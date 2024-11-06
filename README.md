Simple Fastapi example to make inference with tts matxa multispeaker model in onnx format. 

Steps: 

1. Clone the repo.
`git clone https://github.com/langtech-bsc/minimal-tts-multispeaker-api`
2. Build and run the container.
```
cd minimal-tts-multispeaker-api
docker build -t minimal-tts-multispeaker-api .
docker run -p 8000:8000 -t minimal-tts-multispeaker-api
```
3. Test with a simple request.

```
curl -X POST   http://0.0.0.0:8000/api/tts   -H "Content-Type: application/json"   -d '{"text":"Bon dia.","voice":20,"type":"text"}'   | play -t wav -

```
