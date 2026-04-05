# 1. Image de base légère
FROM python:3.11-slim

# 2. Dossier de travail
WORKDIR /app

# 3. Installation des dépendances (Optimisé pour le cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 4. Copie du code source
# On copie tout le dossier src dans /app/src
COPY src/ ./src/

# 5. Copie des artefacts (Modèle + Preprocessing)
# On copie ton dossier local 'artifacts' vers '/app/artifacts' dans l'image
COPY artifacts/ ./artifacts/

# 6. Configuration Python
# On ajoute /app au PYTHONPATH pour que 'src' soit trouvable
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 7. Port FastAPI (8001 car c'est celui que tu as choisi)
EXPOSE 8001

# 8. Lancement de l'application
# On lance le module main.py directement (qui contient uvicorn.run)
CMD ["python", "-m", "src.app.main"]