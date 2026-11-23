from prefect import serve
from prefect.schedules import Cron

from pipeline_prefect import (
    all_flow,
    evaluate_flow,
    import_flow,
    prepare_flow,
    save_flow,
    train_flow,
)

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedule=Cron("0 8 * * *", timezone="Africa/Tunis"),
        ),
        import_flow.to_deployment(name="ml-pipeline-import"),
        prepare_flow.to_deployment(name="ml-pipeline-prepare"),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            schedule=Cron("0 2 * * *", timezone="Africa/Tunis"),
        ),
        save_flow.to_deployment(name="ml-pipeline-save"),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate"),
    )
