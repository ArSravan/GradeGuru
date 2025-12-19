from __future__ import annotations

from flask import Flask, render_template, request

from app.api import api, predict_payload


def _coerce_types(form: dict[str, str]) -> dict:
    """
    Convert HTML form strings into the correct types expected by the model.
    Numbers become ints; everything else stays as string.
    """
    data: dict = dict(form)

    # numeric columns in your features
    numeric_fields = {
        "age", "Medu", "Fedu", "studytime", "failures", "absences",
        "Dalc", "Walc", "health", "G1", "G2"
    }

    # traveltime exists in your model list; include it too
    numeric_fields.add("traveltime")

    for k in list(data.keys()):
        if k in numeric_fields and data[k] != "":
            try:
                data[k] = int(data[k])
            except ValueError:
                # leave as-is if conversion fails; API will error cleanly
                pass

    return data


def create_app() -> Flask:
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
        # Convert form to dict + correct types
        payload = _coerce_types(request.form.to_dict())

        # IMPORTANT: include mode if your form has it; else default to early
        payload.setdefault("mode", "early")

        # Call prediction directly (NO HTTP call)
        output = predict_payload(payload)

        if "error" in output:
            return render_template(
                "result.html",
                predicted_g3="—",
                risk_band="—",
                error_message=output["error"].get("message"),
                missing=output["error"].get("missing", []),
                mode=output.get("mode", "early"),
            ), 400

        return render_template(
            "result.html",
            predicted_g3=output.get("predicted_g3"),
            risk_band=output.get("risk_band"),
            error_message=None,
            missing=[],
            mode=output.get("mode"),
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)