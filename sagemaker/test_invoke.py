import argparse
import json
import boto3


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", type=str, required=True)
    p.add_argument("--text", type=str, default="Hello, world!")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--num-beams", type=int, default=4)
    return p.parse_args()


def main():
    args = parse()
    rt = boto3.client("sagemaker-runtime")

    payload = {
        "inputs": args.text,
        "parameters": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "do_sample": False
        }
    }

    response = rt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = response["Body"].read().decode("utf-8")
    print("Response:", body)


if __name__ == "__main__":
    main()