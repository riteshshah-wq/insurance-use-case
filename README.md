# Insurance AI Use Case

AI-powered insurance workflows on the **GoldenRay AI Platform** — claims processing, underwriting risk assessment, and RAG-powered customer service.

## Use Cases

| Use Case | Description | Endpoint |
|----------|-------------|----------|
| **Claims Processing** | Extract fields from claim documents, classify severity, route to adjusters | `POST /api/claims/process` |
| **Underwriting** | Score applicant risk, calculate premiums with AI narratives | `POST /api/underwriting/premium` |
| **Customer Agent** | RAG-powered chatbot answering policy questions | `POST /api/agent/chat` |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Insurance AI Application (FastAPI)           │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   Claims     │  │ Underwriting │  │ Customer Agent │  │
│  │  Extract     │  │ Risk Score   │  │ RAG + Chat     │  │
│  │  Classify    │  │ Premium Calc │  │                │  │
│  │  Route       │  │              │  │                │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘  │
│         └─────────────────┼──────────────────┘           │
│                    ┌──────▼───────┐                       │
│                    │  LLM Client  │                       │
│                    └──────┬───────┘                       │
└───────────────────────────┼──────────────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │  GoldenRay MaaS  │
                   │  Inference API   │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
         ┌────▼───┐   ┌────▼───┐   ┌─────▼─────┐
         │ Ollama │   │  vLLM  │   │ External  │
         │TinyLlam│   │opt-125m│   │ GPT-4o    │
         └────────┘   └────────┘   └───────────┘
```

## Quick Start

### Run Locally

```bash
git clone https://github.com/riteshshah-wq/insurance-use-case.git
cd insurance-use-case

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure (defaults work with GoldenRay demo cluster)
export GOLDENRAY_API_URL=http://10.0.0.113:30888
export GOLDENRAY_NAMESPACE=demo-tenant
export GOLDENRAY_ENDPOINT=TinyLlama

uvicorn app.api:app --reload --port 8000
```

Visit http://localhost:8000/docs for Swagger UI.

### Deploy to Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/modelendpoint.yaml
kubectl apply -f k8s/guardrails.yaml
```

### Docker

```bash
docker build -t insurance-api .
docker run -p 8000:8000 insurance-api
```

## Example API Calls

### Process a Claim

```bash
curl -X POST http://localhost:8000/api/claims/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Claim #CLM-001. John Smith (AUTO-889921) reports a collision on March 10. Vehicle damage: $15,500. Minor whiplash injury."}'
```

### Calculate Premium

```bash
curl -X POST http://localhost:8000/api/underwriting/premium \
  -H "Content-Type: application/json" \
  -d '{"applicant": {"name": "Emily Rodriguez", "age": 32, "coverage_type": "auto", "coverage_amount": 50000, "driving_record": "clean"}}'
```

### Customer Chat

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my auto insurance deductible?"}'
```

## Project Structure

```
insurance-use-case/
├── app/                        # FastAPI application
│   ├── api.py                  # API routes
│   ├── common/                 # Shared: LLM client, config, models
│   ├── claims/                 # Claims: extract, classify, route
│   ├── underwriting/           # Underwriting: risk, premium
│   └── customer_agent/         # Customer: agent, RAG
├── data/                       # Sample data
│   ├── sample_claims/          # 5 sample claim documents
│   ├── sample_policies/        # Auto, home, life policy docs
│   └── applicant_profiles/     # 3 underwriting test profiles
├── notebooks/                  # Jupyter notebooks
│   ├── 01-data-exploration.ipynb
│   ├── 02-prompt-engineering.ipynb
│   ├── 03-rag-evaluation.ipynb
│   └── 04-model-comparison.ipynb
├── k8s/                        # Kubernetes manifests
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── modelendpoint.yaml      # GoldenRay ModelEndpoint CRD
│   └── guardrails.yaml         # PII protection rules
├── pipelines/                  # Flyte workflows
│   ├── claims_pipeline.py      # Extract → classify → route → MLflow
│   └── training_pipeline.py    # Load → fine-tune → evaluate → register
├── docs/                       # Antora documentation site
│   ├── antora.yml
│   └── modules/
│       ├── ROOT/               # Overview + architecture
│       ├── workshop/           # 5 hands-on labs
│       └── reference/          # API, deployment, model config
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model Configuration

The LLM client works with any model through GoldenRay MaaS — just change the endpoint:

| Model | Command | Quality |
|-------|---------|---------|
| TinyLlama (demo) | `GOLDENRAY_ENDPOINT=TinyLlama` | Basic, demos the flow |
| OPT-125M | `GOLDENRAY_ENDPOINT=facebook/opt-125m` | Minimal, tests infra |
| Llama-3-8B | `GOLDENRAY_ENDPOINT=llama-3-8b` | Production quality |
| GPT-4o (external) | `GOLDENRAY_ENDPOINT=gpt-4o` | Best quality |

No code changes needed. Same API, different model.

## Documentation

Full Antora documentation in `docs/`:

- **Workshop Labs**: Step-by-step hands-on labs (5 labs)
- **API Reference**: Complete endpoint documentation
- **Deployment Guide**: Kubernetes deployment instructions
- **Model Configuration**: Model selection and tuning guide

## License

MIT
