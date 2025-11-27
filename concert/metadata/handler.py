"""
handler.py
----------

Implements functionalities for basic metadata handling.
"""
from datetime import datetime
import json
from typing import Any, Dict, Optional
import requests
from requests import Response
import urllib3
from urllib3.exceptions import SubjectAltNameWarning, InsecureRequestWarning
from concert.metadata.utils import ProposalDTO, DatasetDTO

urllib3.disable_warnings(SubjectAltNameWarning)
urllib3.disable_warnings(InsecureRequestWarning)

class MetadataHandler:
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
    
    def _remote_dispatch_post(self, endpoint: str, payload: Dict[str, Any]) -> Response:
        response = requests.post(
            url=endpoint, headers=self._header,
            json=payload, params=self._params, verify=False)
        if response.status_code == 401:
            self._fetch_token()
            response = requests.post(
                url=endpoint, headers=self._header,
                json=payload, params=self._params, verify=False)
        return response

    def _create_proposal(self, proposal: ProposalDTO) -> Optional[str]:
        endpoint = self._base_endpoint +  "/proposals"
        response = self ._remote_dispatch_post(
            endpoint=endpoint,
            payload=proposal.model_dump(exclude_none=True))
        if response.status_code != 201:
            return None
        return response.json()['proposalId']

    def _create_dataset(self, dataset: DatasetDTO) -> bool:
        endpoint = self._base_endpoint + "/datasets"
        response = self._remote_dispatch_post(
            endpoint=endpoint,
            payload=dataset.model_dump(exclude_none=True))
        if response.status_code != 201:
            return False
        return True

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
        self._proposal_id = self._create_proposal(proposal=proposal)

    def dispatch_dataset(self, ds_name: str, src_dir: str, metadata: Dict[str, Any]) -> bool:
        """
        Dispatches a dataset to data catalog

        :param ds_name: dataset name
        :type ds_name: str
        :param src_dir: source directory of dataset
        :type src_dir: str
        :param metadata: experimental metadata
        :type metadata: str
        """
        dataset = DatasetDTO(
            pid=ds_name, ownerGroup="IPS", principalInvestigator="Tomas Farago",
            owner="Tomas Farago", contactEmail="tomas.farago@kit.edu",
            sourceFolder=src_dir, creationLocation="DESY - PETRA III - P23",
            creationTime=datetime.today().isoformat(), proposalId=self._proposal_id,
            description=f"Serial-MicroCT of {ds_name}", datasetName=ds_name,
            scientificMetadata=metadata
        )
        return self._create_dataset(dataset=dataset)
