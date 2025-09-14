import argparse
import time
import sagemaker
from sagemaker.huggingface import HuggingFace


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--role", type=str, default=None, help="Execution role ARN. If omitted and running in SageMaker, uses get_execution_role().")
    p.add_argument("--bucket", type=str, default=None, help="S3 bucket to store model artifacts; defaults to session.default_bucket().")
    p.add_argument("--region", type=str, default=None, help="AWS region; defaults to the current session's region.")
    p.add_argument("--instance-type", type=str, default="ml.g5.2xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--volume-size", type=int, default=100)
    p.add_argument("--use-spot", action="store_true", help="Use managed spot training")
    p.add_argument("--max-run", type=int, default=3*60*60, help="Max run seconds")
    p.add_argument("--max-wait", type=int, default=None, help="Max wait seconds when using spot (defaults to max-run)")

    # Training hyperparameters (forwarded to entry point)
    p.add_argument("--src-lang", type=str, default="en")
    p.add_argument("--tgt-lang", type=str, default="es")
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--dataset-name", type=str, default="Helsinki-NLP/opus_books")
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=1)
    p.add_argument("--gen-max-len", type=int, default=128)
    p.add_argument("--job-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse()

    sess = sagemaker.Session()
    region = args.region or sess.boto_region_name

    if args.role:
        role = args.role
    else:
        try:
            role = sagemaker.get_execution_role()
        except Exception:
            raise SystemExit("Please pass --role when running outside SageMaker.")

    bucket = args.bucket or sess.default_bucket()
    output_path = f"s3://{bucket}/seq2seq-translation/artifacts"

    # Versions from the public "Available DLCs on AWS"
    # Training: Transformers 4.49.0 on PyTorch 2.5 (Python 3.11)
    estimator = HuggingFace(
        entry_point="sagemaker/train_sagemaker.py",
        source_dir=".",
        role=role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        transformers_version="4.49.0",
        pytorch_version="2.5",
        py_version="py311",
        output_path=output_path,
        use_spot_instances=args.use_spot,
        max_run=args.max_run,
        max_wait=(args.max_wait or args.max_run) if args.use_spot else None,
        hyperparameters={
            "src_lang": args.src_lang,
            "tgt_lang": args.tgt_lang,
            "model_name": args.model_name or "",
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config or "",
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.train_batch_size,
            "per_device_eval_batch_size": args.eval_batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.grad_accum,
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
            "generation_max_length": args.gen_max_len,
        },
    )

    job_name = args.job_name or f"mt-{args.src_lang}-{args.tgt_lang}-{int(time.time())}"
    print(f"Starting training job: {job_name}")
    estimator.fit(job_name=job_name, logs=True)

    print("Done. Model artifacts at:", estimator.model_data)
    print("You can now deploy using: python sagemaker/deploy_endpoint.py --from-training", job_name)


if __name__ == "__main__":
    main()