.PHONY: build run deploy docs test clean

## Build the container image
build:
	podman build -t insurance-use-case .

## Run the app locally with hot-reload
run:
	uvicorn app.api:app --reload --port 8001

## Deploy to Kubernetes
deploy:
	kubectl apply -f k8s/

## Build the Antora documentation site
docs:
	npx antora antora-playbook.yml

## Smoke-test the running API (expects app on localhost:8001)
test:
	@echo "=== Health check ==="
	curl -s http://localhost:8001/health | python3 -m json.tool
	@echo "\n=== Claims processing ==="
	curl -s -X POST http://localhost:8001/api/claims/process \
	  -H "Content-Type: application/json" \
	  -d '{"text": "Claim #TEST-001. Policyholder Jane Doe reports a fender bender on Oct 15, 2024. Bumper damage estimated at $$1,200. No injuries. Policy AUTO-112233."}' \
	  | python3 -m json.tool
	@echo "\n=== Underwriting ==="
	curl -s -X POST http://localhost:8001/api/underwriting/assess \
	  -H "Content-Type: application/json" \
	  -d '{"applicant": {"name": "Test User", "age": 30, "coverage_type": "auto", "coverage_amount": 50000}}' \
	  | python3 -m json.tool

## Remove build artifacts
clean:
	rm -rf build/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
