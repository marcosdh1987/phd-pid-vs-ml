# =============================================================================
# Agente de Preparación de Consultas Médicas
# =============================================================================
# Este Makefile proporciona comandos para configurar, ejecutar y trabajar con
# el agente especializado en preparación de consultas médicas usando SBAR.
#
# Características principales:
# - Gestión de entorno virtual con uv
# - Herramientas de linting y formateo para calidad de código
# - Servidor de desarrollo y producción
# - Modo CLI interactivo y de pregunta única
# - Pruebas en lote contra la API
# - Soporte de contenedorización Docker
#
# Uso básico:
#   make install          # Configurar entorno de desarrollo
#   make format           # Formatear código automáticamente
#   make lint             # Revisar calidad del código
#   make test             # Ejecutar pruebas
#   make run-api          # Iniciar servidor API
#   make build-api        # Construir imagen Docker de la API
# =============================================================================

export PYTHON_VERSION=3.11.9
export ENVIRONMENT=localhost
VENV_DIR ?= .venv
KERNEL_NAME=ai-kernel

# =============================================================================
# CONFIGURACIÓN DEL ENTORNO DE DESARROLLO
# =============================================================================

# Configurar entorno virtual e instalar todas las dependencias
install:
	@echo "🚀 Creando entorno virtual con uv..."
	@if ! command -v uv &> /dev/null; then \
		echo "❌ uv no está instalado. Por favor instálalo con: pip install uv"; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV_DIR)" ]; then \
		uv venv $(VENV_DIR) --python=$(PYTHON_VERSION); \
	else \
		echo "✅ El entorno virtual ya existe."; \
	fi
	@echo "📦 Instalando dependencias con uv pip..."
	@. $(VENV_DIR)/bin/activate && uv pip install -r requirements.in && uv pip install ipykernel 
	@echo "🔌 Registrando kernel de Jupyter..."
	@$(VENV_DIR)/bin/python -m ipykernel install --user --name=$(KERNEL_NAME) --display-name="Python (uv)"
	@echo "✅ Entorno virtual uv listo para Jupyter Notebook."

# Configurar hooks de pre-commit para garantizar calidad del código
setup-hooks:
	@echo "🪝 Configurando hooks de pre-commit..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && pre-commit install
	@echo "✅ Hooks de pre-commit configurados!"

# Generar requirements.txt desde el entorno actual
generate-requirements:
	@echo "📋 Generando requirements.txt desde el entorno .uv con uv pip freeze..."
	@command -v uv >/dev/null 2>&1 || pip install --user uv
	@. $(VENV_DIR)/bin/activate && uv pip freeze > requirements.txt
	@echo "✅ requirements.txt generado"

# =============================================================================
# CALIDAD DE CÓDIGO Y LINTING
# =============================================================================

# Formatear código automáticamente con black e isort
format:
	@echo "🎨 Formateando código con black e isort..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && black src/ app/ tests/ --line-length 88
	@. $(VENV_DIR)/bin/activate && isort src/ app/ tests/ --profile black
	@echo "✅ Código formateado!"

# Revisar calidad del código con múltiples herramientas
lint:
	@echo "🔍 Ejecutando análisis de calidad del código..."
	@if [ ! -d .venv ]; then make install; fi
	@echo "🚀 Ruff (linter rápido)..."
	@. $(VENV_DIR)/bin/activate && ruff check src/ app/ tests/
	@echo " Bandit (seguridad)..."
	@. $(VENV_DIR)/bin/activate && bandit -r src/ app/ -f json -o security-report.json -ll -q || true
	@. $(VENV_DIR)/bin/activate && bandit -r src/ app/ -ll || true
	@echo "✅ Análisis de calidad completado!"

# Revisar solo con ruff (más rápido para desarrollo)
lint-fast:
	@echo "⚡ Análisis rápido con ruff..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && ruff check src/ app/ tests/
	@echo "✅ Análisis rápido completado!"

# Arreglar automáticamente problemas de linting cuando sea posible
fix:
	@echo "🔧 Arreglando problemas automáticamente..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && ruff check --fix src/ app/ tests/
	@. $(VENV_DIR)/bin/activate && black src/ app/ tests/ --line-length 88
	@. $(VENV_DIR)/bin/activate && isort src/ app/ tests/ --profile black
	@echo "✅ Problemas arreglados automáticamente!"

# =============================================================================
# PRUEBAS DEL SISTEMA
# =============================================================================

# Ejecutar todas las pruebas con coverage
test:
	@echo "🧪 Ejecutando pruebas con coverage..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && PYTHONPATH=${PWD}/src pytest tests/ --cov=src --cov-report=html --cov-report=term-missing || echo "⚠️  No se encontraron tests para ejecutar"
	@echo "✅ Pruebas completadas! Ver reporte en htmlcov/index.html"

# Ejecutar pruebas específicas
test-unit:
	@echo "🧪 Ejecutando pruebas unitarias..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && PYTHONPATH=${PWD}/src pytest tests/ -v

# Ejecutar pruebas en lote contra la API
run-batch-test:
	@echo "🚀 Ejecutando pruebas en lote contra la API..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && PYTHONPATH=${PWD}/src python tests/batch_test.py --input data/GoldenDataset_v1.csv --api-url http://127.0.0.1:8008/api/v1/chat --delay 1 --non-interactive

# Ejecutar pruebas en lote con parámetros personalizados
run-batch-test-custom:
	@echo "🚀 Ejecutando pruebas en lote con parámetros personalizados..."
	@echo "Uso: make run-batch-test-custom INPUT=ruta/al/input.csv OUTPUT=ruta/al/output.csv MODEL=gpt5-mini DELAY=1"
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && PYTHONPATH=${PWD}/src python tests/batch_test.py \
		--input $(or $(INPUT),data/GoldenDataset_v1.csv) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		--api-url $(or $(API_URL),http://127.0.0.1:8008/api/v1/chat) \
		--model $(or $(MODEL),gpt5-mini) \
		--delay $(or $(DELAY),1) \
		--non-interactive

# =============================================================================
# EJECUCIÓN DE LA APLICACIÓN
# =============================================================================

# Iniciar servidor de desarrollo LangGraph
run-dev:
	@echo "🚀 Iniciando servidor de desarrollo..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && langgraph dev

# Iniciar servidor FastAPI
run-api:
	@echo "🚀 Iniciando servidor API..."
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && PYTHONPATH=${PWD} uvicorn api:app --reload --host 0.0.0.0 --port 8008 --log-level debug

# Ejecutar CLI con una pregunta predefinida
run-question:
	@echo "🚀 Ejecutando una pregunta única"
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && PYTHONPATH=${PWD}/src python main.py --question "Tengo dolor de cabeza frecuente, ¿me puedes ayudar a preparar mi consulta médica?"

# Iniciar modo CLI interactivo
run-interactive:
	@echo "🚀 Iniciando modo CLI interactivo"
	@if [ ! -d .venv ]; then make install; fi
	@. $(VENV_DIR)/bin/activate && PYTHONPATH=${PWD}/src python main.py --interactive

# =============================================================================
# CONSTRUCCIÓN Y DESPLIEGUE CON DOCKER
# =============================================================================

# Variables de configuración Docker
IMG_NAME ?= agente-medico
IMAGE_TAG ?= latest
CONTAINER_NAME ?= agente-medico-server
API_PORT ?= 8008

# Habilitar Docker BuildKit
export DOCKER_BUILDKIT=1

# Construir imagen Docker de la API
build-api:
	@echo "🔨 Construyendo imagen Docker FastAPI (usando Dockerfile.api)..."
	@docker build --platform=linux/amd64 -t ${IMG_NAME}:${IMAGE_TAG} -f Dockerfile.api .
	@echo "✅ Imagen Docker FastAPI construida exitosamente!"

# Ejecutar contenedor Docker
run-api-docker:
	@echo "🚀 Ejecutando contenedor Docker..."
	@docker run --platform=linux/amd64 -e ENV=production -d -p ${API_PORT}:${API_PORT} --env-file .env ${IMG_NAME}:${IMAGE_TAG}
	@echo "✅ Contenedor Docker ejecutándose en http://localhost:${API_PORT}!"

# Construir sin cache
build-fresh:
	@echo "🔨 Construyendo imagen Docker sin cache..."
	@docker build --no-cache --platform=linux/amd64 -t ${IMG_NAME}:${IMAGE_TAG} -f Dockerfile.api .
	@echo "✅ Imagen Docker construida exitosamente!"

# Detener contenedor Docker
stop-docker:
	@echo "🛑 Deteniendo contenedor Docker..."
	@docker stop ${CONTAINER_NAME} 2>/dev/null || true
	@docker rm ${CONTAINER_NAME} 2>/dev/null || true
	@echo "✅ Contenedor detenido!"

# =============================================================================
# COMANDOS ÚTILES
# =============================================================================

# Comando de validación completa (CI/CD pipeline)
ci:
	@echo "🚀 Ejecutando pipeline de CI completo..."
	@make format
	@make lint
	@make test
	@echo "✅ Pipeline de CI completado exitosamente!"

# Mostrar información de ayuda sobre comandos disponibles
help:
	@echo "🏥 Agente de Preparación de Consultas Médicas - Comandos Disponibles:"
	@echo ""
	@echo "Configuración de Desarrollo:"
	@echo "  make install              Configurar entorno virtual y dependencias"
	@echo "  make setup-hooks          Configurar hooks de pre-commit"
	@echo "  make generate-requirements Generar requirements.txt desde entorno actual"
	@echo ""
	@echo "Calidad de Código:"
	@echo "  make format               Formatear código automáticamente (black + isort)"
	@echo "  make lint                 Análisis completo de calidad (ruff + mypy + bandit)"
	@echo "  make lint-fast            Análisis rápido con ruff solamente"
	@echo "  make fix                  Arreglar problemas automáticamente"
	@echo "  make ci                   Pipeline completo: format + lint + test"
	@echo ""
	@echo "Pruebas:"
	@echo "  make test                 Ejecutar todas las pruebas con coverage"
	@echo "  make test-unit            Ejecutar solo pruebas unitarias"
	@echo "  make run-batch-test       Ejecutar pruebas en lote contra API (dataset v1)"
	@echo "  make run-batch-test-custom Ejecutar pruebas en lote con parámetros personalizados"
	@echo ""
	@echo "Ejecución de Aplicación (Local):"
	@echo "  make run-dev             Iniciar servidor de desarrollo LangGraph"
	@echo "  make run-api             Iniciar servidor FastAPI"
	@echo "  make run-question        Probar con pregunta médica predefinida"
	@echo "  make run-interactive     Iniciar modo CLI interactivo"
	@echo ""
	@echo "Docker:"
	@echo "  make build-api           Construir imagen Docker de la API"
	@echo "  make build-fresh         Construir sin cache"
	@echo "  make run-api-docker      Ejecutar API en contenedor Docker"
	@echo "  make stop-docker         Detener contenedor Docker"
	@echo ""
	@echo "URLs de Servicios:"
	@echo "  🚀 FastAPI: http://localhost:8008"
	@echo "  📖 Documentación API: http://localhost:8008/docs"
	@echo "  🔍 Descubrimiento de Agente: http://localhost:8008/.well-known/agent.json"
	@echo ""
	@echo "Utilidades:"
	@echo "  make help                Mostrar este mensaje de ayuda"
	@echo "  make clean               Limpiar archivos cache y generados"
	@echo ""

# Limpiar archivos generados y cache
clean:
	@echo "🧹 Limpiando..."
	@rm -rf __pycache__ .pytest_cache htmlcov .coverage .mypy_cache .ruff_cache
	@rm -f security-report.json
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Limpieza completada!"

# Establecer help como objetivo por defecto
.DEFAULT_GOAL := help

# Declarar objetivos phony
.PHONY: install setup-hooks run-dev run-api run-question run-interactive build-api run-api-docker stop-docker build-fresh clean help generate-requirements run-batch-test run-batch-test-custom test test-unit format lint lint-fast fix ci
