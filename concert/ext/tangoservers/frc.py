"""
frc.py
-----
Implements a device server to execute Fourier Ring Correlation during acquisition.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Union
import json
import math
import os
import datetime
import numpy as np
from tango import DebugIt
from tango.server import attribute, command, AttrWriteType
import torch
from concert.ext.tangoservers.base import TangoRemoteProcessing, RemoteWalkerMixin
from concert.typing import ArrayLike
from concert.imageprocessing import compute_frc, flat_correct, prepare_frc_state, select_frc_region
from concert.storage import RemoteDirectoryWalker
from concert.helpers import PerformanceTracker


class TangoFourierRingCorrelation(TangoRemoteProcessing, RemoteWalkerMixin):
    """
    Implements Tango device server to compute Fourier Ring Correlation for resolution estimation.
    """

    attr_acq = attribute(
        label="Meta attribute for acquisition",
        dtype=(int,),
        max_dim_x=3,
        access=AttrWriteType.READ_WRITE,
        fget="get_attr_acq",
        fset="set_attr_acq",
        doc="encapsulates acquisition meta information i.e., #darks, #flats, #radios",
    )

    proj_offset = attribute(
        label="Projection offset",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_proj_offset",
        fset="set_proj_offset",
        doc="offset in projections to compute FRC",
    )

    resolution_threshold = attribute(
        label="FRC resolution threshold",
        dtype=str,
        access=AttrWriteType.READ_WRITE,
        fget="get_resolution_threshold",
        fset="set_resolution_threshold",
        doc="Threshold for resolution determination: '1/7', 'half_bit'",
    )

    fluctuation_threshold = attribute(
        label="FRC fluctuation threshold percentage",
        dtype=float,
        access=AttrWriteType.READ_WRITE,
        fget="get_fluctuation_threshold",
        fset="set_fluctuation_threshold",
        doc="Percentage deviation from baseline to flag as fluctuation"
        "(default: 15.0, valid range: 5.0-100.0)",
    )

    crop_height = attribute(
        label="Crop height",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_crop_height",
        fset="set_crop_height",
        doc="Target crop height in pixels for FRC region selection (default: 512)",
    )

    padding_y = attribute(
        label="Vertical padding",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_padding_y",
        fset="set_padding_y",
        doc="Vertical padding in pixels to exclude from top/bottom edges (default: 400)",
    )

    padding_x = attribute(
        label="Horizontal padding",
        dtype=int,
        access=AttrWriteType.READ_WRITE,
        fget="get_padding_x",
        fset="set_padding_x",
        doc="Horizontal padding in pixels to exclude from left/right edges (default: 400)",
    )

    _walker: Optional[RemoteDirectoryWalker]
    _frc_state: Optional[Dict[str, Union[int, torch.Tensor]]]
    _crop_y_start: int
    _crop_y_end: int

    async def init_device(self) -> None:
        await super().init_device()
        self._resolution_threshold = "half_bit"
        self._proj_offset = 1
        self._fluctuation_threshold = 15.0
        self._crop_height = 512
        self._padding_y = 400
        self._padding_x = 400
        self._crop_y_start = 0
        self._crop_y_end = 0
        self._frc_state = None
        self._walker = None
        self.info_stream(
            "%s initialized device with %s, state: %s",
            self.__class__.__name__,
            "CUDA" if torch.cuda.is_available() else "CPU",
            self.get_state(),
        )

    def get_attr_acq(self) -> ArrayLike:
        return self._attr_acq

    def set_attr_acq(self, aa: ArrayLike) -> None:
        self._attr_acq = aa
        self.info_stream(
            "%s: acquisition attributes set to: %s",
            self.__class__.__name__,
            str(self._attr_acq),
        )

    def get_proj_offset(self) -> int:
        return self._proj_offset

    def set_proj_offset(self, offset: int) -> None:
        self._proj_offset = offset
        self.info_stream(
            "%s: proj_offset set to: %s",
            self.__class__.__name__,
            str(self._proj_offset),
        )

    def get_resolution_threshold(self) -> str:
        return self._resolution_threshold

    def set_resolution_threshold(self, threshold: str) -> None:
        self._resolution_threshold = threshold
        self.info_stream(
            "%s: resolution_threshold set to: %s",
            self.__class__.__name__,
            str(self._resolution_threshold),
        )

    def get_fluctuation_threshold(self) -> float:
        return self._fluctuation_threshold

    def set_fluctuation_threshold(self, threshold: float) -> None:
        if threshold < 5.0 or threshold > 100.0:
            raise ValueError(
                f"Fluctuation threshold must be between 5.0 and 100.0 percent, " f"got {threshold}"
            )
        self._fluctuation_threshold = threshold
        self.info_stream(
            "%s: fluctuation_threshold set to: %s",
            self.__class__.__name__,
            f"{self._fluctuation_threshold}%",
        )

    def get_crop_height(self) -> int:
        return self._crop_height

    def set_crop_height(self, height: int) -> None:
        if height < 64 or height > 2048:
            raise ValueError(f"Crop height must be between 64 and 2048 pixels, got {height}")
        self._crop_height = height
        self.info_stream(
            "%s: crop_height set to: %s",
            self.__class__.__name__,
            str(self._crop_height),
        )

    def get_padding_y(self) -> int:
        return self._padding_y

    def set_padding_y(self, padding: int) -> None:
        if padding < 0:
            raise ValueError(f"Padding Y must be non-negative, got {padding}")
        self._padding_y = padding
        self.info_stream(
            "%s: padding_y set to: %s",
            self.__class__.__name__,
            str(self._padding_y),
        )

    def get_padding_x(self) -> int:
        return self._padding_x

    def set_padding_x(self, padding: int) -> None:
        if padding < 0:
            raise ValueError(f"Padding X must be non-negative, got {padding}")
        self._padding_x = padding
        self.info_stream(
            "%s: padding_x set to: %s",
            self.__class__.__name__,
            str(self._padding_x),
        )

    @staticmethod
    def _result_to_json_dict(
        result: Dict[str, Union[ArrayLike, float]],
        proj_i: int,
        proj_j: int,
    ) -> Dict[str, Any]:
        """
        Convert compute_frc() result dict to JSON-serializable dictionary.

        :param result: dictionary returned by compute_frc()
        :param proj_i: first projection index
        :param proj_j: second projection index (proj_i + offset)
        :return: JSON-serializable dictionary
        """

        def _sanitize_for_json(value: float) -> Optional[float]:
            """Convert NaN/Inf to None for JSON compatibility."""
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return value

        return {
            "projection_i": proj_i,
            "projection_j": proj_j,
            "classical_resolution": _sanitize_for_json(float(result["classical_resolution"])),
            "geometric_resolution": _sanitize_for_json(float(result["geometric_resolution"])),
        }

    @staticmethod
    def _compute_outlier_aware_statistics(
        values: List[Optional[float]],
        projection_indices: List[int],
        threshold_percent: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Compute outlier-aware statistics for a resolution metric.

        Identifies projections where resolution deviates significantly from the baseline
        (median) value. Useful for detecting sparse fluctuations in otherwise stable
        tomographic acquisitions.

        :param values: list of resolution values (may contain None for NaN/invalid)
        :param projection_indices: corresponding projection_i indices for each value
        :param threshold_percent: percentage deviation from baseline to flag as fluctuation
        :return: dictionary with baseline, counts, and fluctuation details
        """
        # Filter out None/NaN values
        valid_pairs = [(v, i) for v, i in zip(values, projection_indices) if v is not None]

        if not valid_pairs:
            return {
                "baseline_median": None,
                "num_measurements": len(values),
                "num_valid": 0,
                "num_stable": 0,
                "num_fluctuations": 0,
                "fluctuation_threshold_percent": threshold_percent,
                "fluctuation_indices": [],
                "fluctuation_details": [],
            }

        # Extract just the values for median calculation
        valid_values = [v for v, i in valid_pairs]

        # Compute baseline (median)
        baseline_median = float(np.median(valid_values))

        # Compute absolute threshold
        threshold_absolute = baseline_median * (threshold_percent / 100.0)

        # Identify fluctuations
        fluctuation_indices = []
        fluctuation_details = []

        for value, proj_idx in valid_pairs:
            deviation = value - baseline_median
            deviation_percent = (deviation / baseline_median) * 100.0

            if abs(deviation) > threshold_absolute:
                fluctuation_indices.append(proj_idx)
                fluctuation_details.append(
                    {
                        "projection_i": proj_idx,
                        "value": round(value, 4),
                        "deviation_from_baseline": round(deviation, 4),
                        "deviation_percent": round(deviation_percent, 2),
                    }
                )

        num_stable = len(valid_pairs) - len(fluctuation_details)

        return {
            "baseline_median": round(baseline_median, 4),
            "num_measurements": len(values),
            "num_valid": len(valid_pairs),
            "num_stable": num_stable,
            "num_fluctuations": len(fluctuation_details),
            "fluctuation_threshold_percent": threshold_percent,
            "fluctuation_indices": fluctuation_indices,
            "fluctuation_details": fluctuation_details,
        }

    async def _process_flat_fields(self, name: str, producer: AsyncIterator[ArrayLike]) -> None:
        """
        Accumulates dark and flat fields, average them and store as dynamic class members, _dark,
        _flat
        """
        num_proj = 0
        buffer: List[ArrayLike] = []
        async for projection in producer:
            buffer.append(projection)
            num_proj += 1
        setattr(self, name, np.array(buffer).mean(axis=0))
        self.info_stream(
            "%s: processed %d %s projections",
            self.__class__.__name__,
            num_proj,
            name[1:],
        )

    @DebugIt()
    @command()
    async def update_darks(self) -> None:
        await self._process_stream(self._process_flat_fields("_dark", self._receiver.subscribe()))

    @DebugIt()
    @command()
    async def update_flats(self) -> None:
        await self._process_stream(self._process_flat_fields("_flat", self._receiver.subscribe()))

    @DebugIt()
    @command(dtype_in=str)
    async def estimate_spatial_resolution(self, path: str) -> None:
        """Computes FRC for resolution estimation from radiogram projections"""
        await self._process_stream(
            self._estimate_spatial_resolution(self._receiver.subscribe(), path)
        )

    async def _estimate_spatial_resolution(
        self, producer: AsyncIterator[ArrayLike], path: str
    ) -> None:
        """
        Compute Fourier Ring Correlation between consecutive projections with configurable offset.
        Results are stored as JSON file in specified path.

        :param producer: asynchronous generator of projections.
        :type producer: AsyncIterator[ArrayLike]
        :param path: directory where results will be saved
        :type path: str
        """
        correlation_results: List[Dict[str, Any]] = []
        classical_resolutions: List[Optional[float]] = []
        geometric_resolutions: List[Optional[float]] = []
        proj_indices: List[int] = []
        proj_idx: int = 0
        previous_proj: Optional[ArrayLike] = None
        crop_determined = False
        with PerformanceTracker():
            try:
                async for proj in producer:
                    # Determine crop region from first projection.
                    if not crop_determined:
                        fc_proj = flat_correct(proj, self._flat, self._dark)
                        y_start, y_end, crop_method = select_frc_region(
                            fc_proj,
                            crop_height=self._crop_height,
                            padding_y=self._padding_y,
                            padding_x=self._padding_x,
                        )
                        self._crop_y_start = y_start
                        self._crop_y_end = y_end
                        crop_determined = True
                        self.info_stream(
                            "FRC crop region selected: y=[%d:%d] (method=%s)",
                            y_start,
                            y_end,
                            crop_method,
                        )
                    # Initialize state for FRC by computing the frequency beans.
                    if not self._frc_state:
                        self._frc_state = prepare_frc_state(self._crop_height, proj.shape[1])
                        self.info_stream(
                            "FRC frequency bins prepared and state initialized for dim: [%d x %d]",
                            self._crop_height,
                            proj.shape[1],
                        )
                    # When proj_idx = 0 i.e. we are processing the very first projection, previous
                    # projection is also None. In that case we will only store current projection
                    # for correlating later on.
                    if previous_proj is not None:
                        proj_pair_start_idx = proj_idx - 1
                        # Start correlating if offset is satisfied
                        if proj_pair_start_idx % self._proj_offset == 0:
                            self.info_stream(
                                "%s: correlating: proj: %d to proj: %d",
                                self.__class__.__name__,
                                proj_pair_start_idx,
                                proj_pair_start_idx + 1,
                            )
                            result = compute_frc(
                                previous_proj[self._crop_y_start : self._crop_y_end, :],
                                proj[self._crop_y_start : self._crop_y_end, :],
                                self._dark[self._crop_y_start : self._crop_y_end, :],
                                self._flat[self._crop_y_start : self._crop_y_end, :],
                                frc_state=self._frc_state,
                                threshold_method=self._resolution_threshold,
                            )
                            json_result = self._result_to_json_dict(
                                result, proj_idx - self._proj_offset, proj_idx
                            )
                            correlation_results.append(json_result)
                            # Collect for statistics
                            classical_resolutions.append(json_result["classical_resolution"])
                            geometric_resolutions.append(json_result["geometric_resolution"])
                            proj_indices.append(proj_idx - self._proj_offset)
                    previous_proj = proj
                    proj_idx += 1

                # Compute outlier-aware statistics
                classical_stats = self._compute_outlier_aware_statistics(
                    classical_resolutions,
                    proj_indices,
                    threshold_percent=self._fluctuation_threshold,
                )

                geometric_stats = self._compute_outlier_aware_statistics(
                    geometric_resolutions,
                    proj_indices,
                    threshold_percent=self._fluctuation_threshold,
                )

                summary_statistics = {
                    "classical_resolution": classical_stats,
                    "geometric_resolution": geometric_stats,
                }

                # Build final JSON structure.
                # At this point the absolute path looks like ../acq_id/radios because we passed in the
                # current directory, which walker points to. Hence after splitting we need to grab the
                # second last component.
                acquisition_id = os.path.normpath(path).split("/")[-2]
                acquisition_path = "/".join(os.path.normpath(path).split("/")[:-1])
                payload = {
                    "acquisition_id": acquisition_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
                    "configuration": {
                        "offset": self._proj_offset,
                        "threshold_method": self._resolution_threshold,
                        "num_projections_received": proj_idx,
                        "num_frc_computations": len(correlation_results),
                        "crop_y_start": self._crop_y_start,
                        "crop_y_end": self._crop_y_end,
                    },
                    "summary_statistics": summary_statistics,
                    "correlation_results": correlation_results,
                }

                # Construct full file path
                json_filepath = os.path.join(acquisition_path, f"{acquisition_id}_frc.json")

                # Write JSON file
                try:
                    assert self._walker
                    await self._walker.log_to_json(
                        payload=json.dumps(payload, indent=4), filename=json_filepath
                    )
                    self.info_stream(
                        "Saved FRC results to %s (%d measurements)",
                        json_filepath,
                        len(correlation_results),
                    )
                except Exception as e:
                    self.error_stream("Failed to write FRC JSON: %s. Continuing anyway.", str(e))

                if crop_determined:
                    self.info_stream(
                        "FRC completed with crop: y=[%d:%d] (method=%s)",
                        self._crop_y_start,
                        self._crop_y_end,
                        "variance" if self._crop_y_start != 0 else "center",
                    )
                else:
                    self.info_stream("FRC completed without cropping (no projections)")

            except Exception as e:
                self.error_stream("FRC computation failed: %s", str(e))
                raise