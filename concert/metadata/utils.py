"""
utils.py
--------

Encapsulates utilities for basic metadata handling.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel # type: ignore

class CommonDTO(BaseModel):
    """Encapsulates common DTO fields"""

    ownerGroup: str
    accessGroups: Optional[List[str]] = None
    startTime: Optional[str] = None
    endTime: Optional[str] = None
    isPublished: Optional[bool] = None
    instrumentIds: Optional[List[str]] = None
    createdBy: Optional[str] = None
    updatedBy: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    _id: Optional[str] = None
    __v: Optional[int] = None # type: ignore


class ProposalDTO(CommonDTO):
    """Encapsulates DTO fields for proposal"""

    proposalId: str
    parentProposalId: Optional[str] = None
    type: str = "Default Proposal"  # Mandatory field-value
    pi_firstname: Optional[str] = None
    pi_lastname: Optional[str] = None
    pi_email: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    email: str
    title: str
    abstract: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    MeasurementPeriodList: Optional[List[str]] = None
    id: Optional[str] = None


class DatasetDTO(CommonDTO):
    """Encapsulates DTO fields for dataset"""

    pid: Optional[str] = None
    type: str = "raw"  # Mandatory field-value
    principalInvestigator: str
    owner: str
    ownerEmail: Optional[str] = None
    contactEmail: str
    sourceFolder: str
    sourceFolderHost: Optional[str] = None
    numberOfFiles: Optional[int] = None
    creationTime: str
    keywords: Optional[List[str]] = None
    description: Optional[str] = None
    datasetName: Optional[str] = None
    scientificMetadata: Optional[Dict[str, Any]] = None
    scientificMetadataSchema: Optional[str] = None
    dataQualityMetrics: Optional[int] = None
    creationLocation: str
    proposalId: Optional[str] = None
    usedSoftware: Optional[List[str]] = None
    instrumentGroup: Optional[str] = None
    orcidOfOwner: Optional[str] = None
    size: Optional[int] =  None
    packedSize: Optional[int] = None
    numberOfFilesArchived: Optional[int] = None
    validationStatus: Optional[str] = None
    classification: Optional[str] = None
    license: Optional[str] = None
    techniques: Optional[List[str]] = None
    sharedWith: Optional[List[str]] = None
    relationships: Optional[List[str]] = None
    datasetlifecycle: Optional[Dict[str, Any]] = None
    comment: Optional[str] = None
    principalInvestigators: Optional[List[str]] = None
    dataFormat: Optional[str] = None
    proposalIds: Optional[List[str]] = None
    instrumentId: Optional[str] = None
    sampleId: Optional[str] = None
    sampleIds: Optional[List[str]] = None
    inputDatasets: Optional[List[str]] = None
    jobParameters: Optional[Dict[str, Any]] = None
    jobLogData: Optional[str] = None
    runNumber: Optional[str] = None
    version: Optional[str] = None
    history: Optional[List[str]] = None
