import base64
import hashlib
import os
from email.utils import formatdate
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MOCK_DOMAIN: str = os.getenv("MOCK_DOMAIN", "mock-instance")
ACTOR_ID: str = f"http://{MOCK_DOMAIN}/actor"
KEY_ID: str = f"{ACTOR_ID}#main-key"

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_public_key = _private_key.public_key()

PUBLIC_KEY_PEM: str = _public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()

app = FastAPI(title="Mock ActivityPub Instance", version="0.1.0")


@app.get("/.well-known/webfinger")
def webfinger(resource: str = Query(...)):
    expected = f"acct:hub@{MOCK_DOMAIN}"
    if resource != expected:
        raise HTTPException(status_code=404, detail=f"Unknown resource: {resource}")

    return JSONResponse(
        content={
            "subject": resource,
            "links": [
                {
                    "rel": "self",
                    "type": "application/activity+json",
                    "href": ACTOR_ID,
                }
            ],
        },
        media_type="application/jrd+json",
    )


@app.get("/actor")
def actor():
    return JSONResponse(
        content={
            "@context": [
                "https://www.w3.org/ns/activitystreams",
                "https://w3id.org/security/v1",
            ],
            "id": ACTOR_ID,
            "type": "Person",
            "preferredUsername": "hub",
            "inbox": f"http://{MOCK_DOMAIN}/inbox",
            "publicKey": {
                "id": KEY_ID,
                "owner": ACTOR_ID,
                "publicKeyPem": PUBLIC_KEY_PEM,
            },
        },
        media_type="application/activity+json",
    )


class SignRequestBody(BaseModel):
    method: str
    url: str
    body: str = ""


@app.post("/sign-request")
def sign_request(data: SignRequestBody):
    method = data.method.lower()
    parsed = urlparse(data.url)
    host = parsed.netloc or MOCK_DOMAIN
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    date = formatdate(usegmt=True)

    body_bytes = data.body.encode("utf-8") if data.body else b""
    digest = "SHA-256=" + base64.b64encode(hashlib.sha256(body_bytes).digest()).decode()

    signing_string = (
        f"(request-target): {method} {path}\n"
        f"host: {host}\n"
        f"date: {date}\n"
        f"digest: {digest}"
    )

    signature_b64 = base64.b64encode(
        _private_key.sign(signing_string.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
    ).decode()

    return {
        "headers": {
            "Host": host,
            "Date": date,
            "Digest": digest,
            "Signature": (
                f'keyId="{KEY_ID}",'
                f'algorithm="rsa-sha256",'
                f'headers="(request-target) host date digest",'
                f'signature="{signature_b64}"'
            ),
        }
    }
