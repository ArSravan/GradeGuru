"""
GradeGuru Flask Application Entry Point.

This module defines the Flask application factory and web routes for the
GradeGuru student academic performance prediction system.

Responsibilities:
- Create and configure the Flask application instance
- Register the REST API blueprint
- Serve HTML pages for the web interface
- Handle form submissions and invoke the ML prediction pipeline directly
- Perform lightweight input type coercion for HTML form data

Key Design Notes:
- Uses the application factory pattern via `create_app()` for flexibility
  and deployment compatibility (WSGI, Docker, local runs).
- Prediction requests from the web UI do NOT make HTTP calls; they directly
  call the internal `predict_payload` function for efficiency and simplicity.
- Input validation and error reporting are delegated to the API layer.
- Model artifacts must already exist in the `models/` directory for inference.

Routes:
- GET  /            → Home page
- GET  /dashboard   → Feature input form
- POST /result      → Prediction result page

Execution:
- Intended to be run via `python wsgi.py` or through a WSGI server.
- When executed directly, starts a development server on 0.0.0.0:5000.

This module does not train models. It assumes trained models are available.
"""

from __future__ import annotations

from flask import Flask, render_template, request

from app.api import api, predict_payload


def _coerce_types(form: dict[str, str]) -> dict:
    """
    Coerce HTML form values into the types expected by the ML pipeline.

    HTML form submissions provide values as strings. This helper converts known
    numeric feature fields to integers when possible and leaves all other
    values unchanged.

    Conversion rules:
    - Fields listed in `numeric_fields` are converted to `int` if non-empty.
    - If conversion fails (e.g., non-numeric input), the original string is
      preserved so that downstream validation can return a clear error message.

    Parameters
    ----------
    form:
        A mapping of form field names to string values, typically produced by
        `request.form.to_dict()`.

    Returns
    -------
    dict
        A shallow copy of the input mapping with numeric fields converted to
        integers where possible.
    """
    data: dict = dict(form)

    numeric_fields = {
        "age",
        "Medu",
        "Fedu",
        "studytime",
        "failures",
        "absences",
        "Dalc",
        "Walc",
        "health",
        "G1",
        "G2",
        "traveltime",
    }

    for key in list(data.keys()):
        if key in numeric_fields and data[key] != "":
            try:
                data[key] = int(data[key])
            except ValueError:
                # Keep original value; API validation will surface a clean error.
                pass

    return data


def create_app() -> Flask:
    """
    Application factory for GradeGuru.

    Creates a Flask application instance, registers the API blueprint, and sets
    up web routes for the HTML interface.

    Returns
    -------
    flask.Flask
        Configured Flask app instance.
    """
    app = Flask(__name__)
    app.register_blueprint(api)

    @app.get("/")
    def home():
        """Render the landing page."""
        return render_template("index.html")

    @app.get("/dashboard")
    def dashboard():
        """Render the dashboard form page for feature input."""
        return render_template("dashboard.html")

    @app.post("/result")
    def result():
        """
        Handle form submission, run prediction, and render the results page.

        Steps:
        - Convert submitted form fields to a payload dict
        - Coerce numeric features to integers where applicable
        - Ensure a `mode` is present (defaults to "early")
        - Invoke `predict_payload` directly (no internal HTTP request)
        - Render result page on success, or error details on failure

        Returns
        -------
        flask.Response
            Rendered HTML response. On validation/prediction error, returns
            an error view with HTTP 400.
        """
        payload = _coerce_types(request.form.to_dict())
        payload.setdefault("mode", "early")

        output = predict_payload(payload)

        if "error" in output:
            return (
                render_template(
                    "result.html",
                    predicted_g3="—",
                    risk_band="—",
                    error_message=output["error"].get("message"),
                    missing=output["error"].get("missing", []),
                    mode=output.get("mode", "early"),
                ),
                400,
            )

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
