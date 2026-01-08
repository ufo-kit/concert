"""
Docstring for concert.metadata.strategies
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
import json
import requests
from requests import Response
import urllib3
from urllib3.exceptions import SubjectAltNameWarning, InsecureRequestWarning
from concert.storage import Walker
from concert.metadata.utils import ProposalDTO, DatasetDTO

urllib3.disable_warnings(SubjectAltNameWarning)
urllib3.disable_warnings(InsecureRequestWarning)


class MetadataHandlingError(RuntimeError):
    """Defines error type for remote metadata handling"""
    pass


class MetadataHandler(ABC):
    """
    Abstract metadata handler
    """

    @abstractmethod
    async def handle(self, metadata: str, **kwargs) -> None:
        """
        Persists metadata using underlying handling strategy.
        
        :param metadata: metadata
        :type metadata: str
        """
        pass


class FileHandler(MetadataHandler):
    """
    Docstring for FileHandler
    """

    _walker: Walker

    def __init__(self, walker: Walker) -> None:
        self._walker = walker

    async def handle(self, metadata: str, **kwargs) -> None:
        filename = kwargs.get("filename", "experiment.json")
        await self._walker.log_to_json(payload=metadata, filename=filename)
    

class DatabaseHandler(MetadataHandler):
    """Dispatches acquisition metadata payload to a remote endpoint"""

    _base_endpoint: str
    _secret: str
    _token: str
    _proposal_id: Optional[str]

    def __init__(self, endpoint: str, secret: str = ".secret") -> None:
        """
        Initializes the metadata handler.

        :param endpoint: api endpoint for data cataloging
        :type endpoint: str
        :param secret: credentials for data cataloging
        :type secret: str
        """
        self._base_endpoint = endpoint
        self._secret = secret
        self._proposal_id = None
        self._fetch_token()

    def _fetch_token(self) -> None:
        endpoint = self._base_endpoint +  "/auth/login"
        with open(file=self._secret, mode="r") as sf:
            auths = sf.readlines()
            user = auths[0].rstrip().split("=")[1]
            password = auths[1].rstrip().split("=")[1]
            response = requests.post(
                endpoint, json={"username": user, "password": password},
                headers={}, stream=False, verify=False)
            if not response.ok:
                try:
                    response_text = response.json()
                except json.decoder.JSONDecodeError:
                    response_text = response.text
                    raise RuntimeError(f"auth failure: {response_text}")
            self._token = response.json()["id"]

    @property
    def _header(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}
    
    @property
    def _params(self) -> Dict[str, str]:
        return {"access_token": self._token}
    
    @property
    def _get_timestamp(self) -> str:
        return "{:%Y-%m-%d_%H:%M:%S}".format(datetime.now())
    
    def _remote_dispatch_post(self, endpoint: str, payload: Dict[str, Any]) -> Response:
        response = requests.post(
            url=endpoint, headers=self._header,
            json=payload, params=self._params, verify=False)
        if response.status_code in [401, 403]:
            self._fetch_token()
            print("Catalog app auth token refreshed")
            response = requests.post(
                url=endpoint, headers=self._header,
                json=payload, params=self._params, verify=False)
        return response

    def _create_proposal(self, proposal: ProposalDTO) -> Tuple[bool, str]:
        endpoint = self._base_endpoint +  "/proposals"
        response = self ._remote_dispatch_post(
            endpoint=endpoint,
            payload=json.loads(proposal.json(exclude_none=True)))  # BaseModel.json is deprecated
        if response.status_code != 201:
            return False, response.text
        return True, response.json()['proposalId']

    def _create_dataset(self, dataset: DatasetDTO) -> Tuple[bool, str]:
        endpoint = self._base_endpoint + "/datasets"
        response = self._remote_dispatch_post(
            endpoint=endpoint,
            payload=json.loads(dataset.json(exclude_none=True)))  # BaseModel.json is deprecated
        if response.status_code != 201:
            return False, response.text
        return True, ""

    def register_proposal(self, pid: str, title: str) -> None:
        """
        Registers a simple proposal with data catalog

        :param pid: identifier for proposal
        :type pid: str
        :param title: title of the proposal
        :type title: str
        """
        proposal = ProposalDTO(
            ownerGroup="IPS", proposalId=pid, firstname="Tomas",
            lastname="Farago", email="tomas.farago@kit.edu", title=title)
        ok, response = self._create_proposal(proposal=proposal)
        if not ok:
            raise MetadataHandlingError(response)
        self._proposal_id = response

    async def handle(self, metadata: str, **kwargs) -> None:
        """
        Dispatches a dataset to data catalog

        :param ds_name: dataset name
        :type ds_name: str
        :param src_dir: source directory of dataset
        :type src_dir: str
        :param metadata: experimental metadata
        :type metadata: str
        """
        ds_name = kwargs.get("ds_name", "generic_raw_dataset")
        src_dir = kwargs.get("src_dir", "")
        dataset = DatasetDTO(
            pid=f"{ds_name}-{self._get_timestamp}", ownerGroup="IPS",
            principalInvestigator="Tomas Farago", owner="Tomas Farago",
            contactEmail="tomas.farago@kit.edu", sourceFolder=src_dir,
            creationLocation="DESY - PETRA III - P23", creationTime=datetime.now().isoformat(),
            proposalId=self._proposal_id, description=f"Serial-MicroCT Dataset - {ds_name}",
            datasetName=ds_name, scientificMetadata=json.loads(metadata)
        )
        self._create_dataset(dataset=dataset)