import argparse
import sagemaker
from sagemaker.huggingface import HuggingFace, HuggingFaceModel


def parse():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-training", type=str, help="Existing training job name to attach")
    src.add_argument("--model-data", type=str, help="S3 URI to model.tar.gz")

    p.add_argument("--endpoint-name", type=str, required=True)
    p.add_argument("--instance-type", type=str, default="ml.m5.xlarge")
    p.add_argument("--initial-instance-count", type=int, default=1)
    p.add_argument("--role", type=str, default=None, help="Execution role ARN. If omitted and running in SageMaker, uses get_execution_role().")
    p.add_argument("--use-default-pipeline", action="store_true", help="Use HF_TASK=translation instead of custom inference handler.")
    return p.parse_args()


def main():
    args = parse()
    sess = sagemaker.Session()

    if args.role:
        role = args.role
    else:
        try:
            role = sagemaker.get_execution_role()
        except Exception:
            raise SystemExit("Please pass --role when running outside SageMaker.")

    if args.from_training:
        # Attach to the completed training job
        estimator = HuggingFace.attach(args.from_training, sagemaker_session=sess)
        model_data = estimator.model_data
    else:
        model_data = args.model_data

    if args.use_default_pipeline:
        # Simpler: let the container create a translation pipeline
        hf_model = HuggingFaceModel(
            role=role,
            model_data=model_data,
            transformers_version="4.51.3",
            pytorch_version="2.6",
            py_version="py312",
            env={"HF_TASK": "translation"},
        )
        predictor = hf_model.deploy(
            initial_instance_count=args.initial_instance_count,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
        )
    else:
        # Use our custom inference handler (sagemaker/inference.py)
        model = estimator.create_model(
            role=role,
            entry_point="inference.py",
            source_dir="sagemaker",
            transformers_version="4.51.3",
            pytorch_version="2.6",
            py_version="py312",
        )
        predictor = model.deploy(
            initial_instance_count=args.initial_instance_count,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
        )

    print("✅ Deployed endpoint:", args.endpoint_name)
    print("Invoke with: python sagemaker/test_invoke.py --endpoint-name", args.endpoint_name)


if __name__ == "__main__":
    main()