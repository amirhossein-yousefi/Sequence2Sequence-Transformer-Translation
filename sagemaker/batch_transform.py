import argparse
import sagemaker
from sagemaker.huggingface import HuggingFaceModel


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model-data", type=str, required=True, help="S3 URI of model.tar.gz.")
    p.add_argument("--input", type=str, required=True, help="S3 URI to .jsonl input file.")
    p.add_argument("--output", type=str, required=True, help="S3 URI to output folder.")
    p.add_argument("--instance-type", type=str, default="ml.m5.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--role", type=str, default=None, help="Execution role ARN. If omitted and running in SageMaker, uses get_execution_role().")
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

    hf_model = HuggingFaceModel(
        role=role,
        model_data=args.model_data,
        transformers_version="4.51.3",
        pytorch_version="2.6",
        py_version="py312",
        env={"HF_TASK": "translation"},
    )

    transformer = hf_model.transformer(
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        strategy="SingleRecord",
        output_path=args.output,
    )

    transformer.transform(
        data=args.input,
        content_type="application/json",
        split_type="Line",
    )

    transformer.wait()
    print("Batch transform complete. Results written to:", args.output)


if __name__ == "__main__":
    main()