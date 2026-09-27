"""Recorded /v1/systemone responses, the questions they answer, and a mock HTTP handler for unit tests."""

import json
from pathlib import Path

import httpx
from lattereview.providers import Noul, Choice, Score

RESPONSES = json.loads((Path(__file__).parent / "fixtures" / "systemone_responses.json").read_text())

# The questions the recorded responses answer.
REQUEST_QUESTIONS = {
    "dl": Noul("Does the study use deep learning?", true="uses deep learning", false="does not"),
    "mod": Choice(
        "Which imaging modality is studied?",
        {"CT": "computed tomography", "MRI": "magnetic resonance", "XR": "radiography", "US": "ultrasound"},
    ),
    "q": Score("How strong is the validation?", ["none", "internal only", "external"]),
    "plain": Noul("Is this a randomized controlled trial?"),
}


class Recorder:
    """An httpx mock handler that replays queued responses and records the requests it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(request)
        return response

    @property
    def bodies(self):
        return [json.loads(request.content) for request in self.requests]
