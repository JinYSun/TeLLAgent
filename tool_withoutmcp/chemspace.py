import os
import pandas as pd
import requests
from langchain.tools import BaseTool

from utils import is_smiles


class ChemSpace:
    def __init__(self, chemspace_api_key=None):
        self.chemspace_api_key = chemspace_api_key
        self._renew_token()  # Create token

    def _renew_token(self):
        self.chemspace_token = requests.get(
            url="https://api.chem-space.com/auth/token",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.chemspace_api_key}",
            },
        ).json()["access_token"]
            
    def _make_api_request(
        self,
        query,
        request_type,
        count,
        categories,
    ):
        """
        Make a generic request to chem-space API.

        Categories request.
            CSCS: Custom Request: Could be useful for requesting whole synthesis
            CSMB: Make-On-Demand Building Blocks
            CSSB: In-Stock Building Blocks
            CSSS: In-stock Screening Compounds
            CSMS: Make-On-Demand Screening Compounds
        """

        def _do_request():
            data = requests.request(
                "POST",
                url=f"https://api.chem-space.com/v3/search/{request_type}?count={count}&page=1&categories={categories}",
                headers={
                    "Accept": "application/json; version=3.1",
                    "Authorization": f"Bearer {self.chemspace_token}",
                },
                data={"SMILES": f"{query}"},
            ).json()
            return data

        data = _do_request()

        # renew token if token is invalid
        if "message" in data.keys():
            if data["message"] == "Your request was made with invalid credentials.":
                self._renew_token()

        data = _do_request()
        return data

    def _convert_single(self, query, search_type: str):
        """Do query for a single molecule"""
        data = self._make_api_request(query, "exact", 1, "CSCS,CSMB,CSSB")
        if data["count"] > 0:
            return data["items"][0][search_type]
        else:
            return "No data was found for this compound."

    def convert_mol_rep(self, query, search_type: str = "smiles"):
        if ", " in query:
            query_list = query.split(", ")
        else:
            query_list = [query]
        smi = ""
        try:
            for q in query_list:
                smi += f"{query}'s {search_type} is: {str(self._convert_single(q, search_type))}"
                return smi
        except Exception:
            return "The input provided is wrong. Input either a single molecule, or multiple molecules separated by a ', '"
