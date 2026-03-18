# Document Crop API

API que recorta documentos (DNI, tarjetas, pasaportes) de fotos usando OpenCV.

## Deploy en Railway

1. Crea un repo en GitHub con estos archivos (`app.py`, `requirements.txt`, `Procfile`)
2. Ve a [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Selecciona el repo → Railway lo despliega automáticamente
4. En **Settings** → **Networking** → **Generate Domain** (para tener URL pública)
5. Copia la URL (ej: `https://tu-servicio.up.railway.app`)

## Uso

### Desde n8n (con URL de imagen)

Nodo **HTTP Request**:
- **Method**: POST
- **URL**: `https://tu-servicio.up.railway.app/crop`
- **Body Type**: JSON
- **Body**: `{ "image_url": "{{ $json.image_url }}" }`
- **Response Format**: File

La respuesta es la imagen PNG recortada como binario → pasa directo al nodo de Google Drive Upload.

### Desde n8n (con imagen binaria)

Nodo **HTTP Request**:
- **Method**: POST
- **URL**: `https://tu-servicio.up.railway.app/crop`
- **Body Type**: Binary
- **Input Data Field Name**: `data`
- **Response Format**: File

## Flujo en n8n

```
... → HTTP Request (descargar imagen) → HTTP Request (POST /crop) → Google Drive (upload) → ...
```

## Health check

```
GET https://tu-servicio.up.railway.app/
→ {"status": "ok", "service": "document-crop"}
```
