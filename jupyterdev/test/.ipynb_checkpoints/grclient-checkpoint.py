from gradio_client import Client

client = Client("http://localhost:7860/")
result = client.predict(
		api_name="/_update_display"
)
print(result)
