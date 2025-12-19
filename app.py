import requests
from flask import Flask, render_template,session,request
from app.api import api


def create_app():
    app = Flask(__name__)   
    app.register_blueprint(api)

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    

    @app.post("/result")
    def result():
        data = request.form

        response = requests.post(
            "http://127.0.0.1:5000/api/predict",
            json=data
        ).json()

        return render_template(
            "result.html",
            predicted_g3=response.get("predicted_g3"),
            risk_band=response.get("risk_band"),
    )
    return app


if __name__ == "__main__":   
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)