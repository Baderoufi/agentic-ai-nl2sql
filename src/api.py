# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# import core pipeline
from .nl2sql_core import nl2sql_with_acl


app = FastAPI(
    title="NL2SQL with ACL API",
    description="MySQL NL2SQL service with ACL + ChromaDB + Groq LLM",
    version="1.0.0",
)

# ====== Request / Response Models ======

class NL2SQLRequest(BaseModel):
    user_id: str
    question: str
    n_rows: int = 50
    strict_acl: bool = True

class NL2SQLResponse(BaseModel):
    sql: str

class ErrorResponse(BaseModel):
    detail: str


# ====== Health Check ======

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ====== Main Endpoint ======

@app.post(
    "/nl2sql",
    response_model=NL2SQLResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def generate_sql(payload: NL2SQLRequest):
    try:
        sql = nl2sql_with_acl(
            user_id=payload.user_id,
            natural_language_query=payload.question,
            n_rows=payload.n_rows,
            strict_acl=payload.strict_acl,
        )
        return NL2SQLResponse(sql=sql)
    except Exception as e:
        # حالياً نرجع 400 على كل الأخطاء؛
        # تقدر تفرّق لاحقاً بين ACL / LLM / DB لو حبيت.
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
