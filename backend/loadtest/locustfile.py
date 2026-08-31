from __future__ import annotations

from locust import (
    HttpUser,
    between,
    task,
)


class ECProUser(
    HttpUser
):
    """
    Simulated EC Pro inference user.
    """

    wait_time = between(
        0.5,
        2.0,
    )

    @task
    def chat(
        self,
    ) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain how "
                        "EC Pro should "
                        "scale under "
                        "production load."
                    ),
                }
            ],
            "max_new_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        with self.client.post(
            "/chat",
            json=payload,
            name="/chat",
            catch_response=True,
        ) as response:
            if (
                response.status_code
                == 200
            ):
                response.success()

                return

            if (
                response.status_code
                == 503
            ):
                response.failure(
                    "Inference service "
                    "was not ready"
                )

                return

            response.failure(
                "Unexpected HTTP "
                "status "
                f"{response.status_code}"
            )