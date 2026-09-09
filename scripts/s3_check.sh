#!/usr/bin/env bash
# Resolve a bucket's REAL region and check it against the one we are about to use.
#
# Why this exists: AWS region comes from several places that disagree silently —
# the AWS_REGION environment variable, AWS_DEFAULT_REGION, the profile's `region`
# in ~/.aws/config, and the Makefile default. Make inherits environment variables,
# so a `?=` default is overridden without a word. Meanwhile `aws s3` often still
# works against the wrong region because S3 redirects, so the mistake shows up as
# unexplained slowness and cross-region transfer charges rather than an error.
#
# The bucket itself is the only authority. Ask it.
#
#   scripts/s3_check.sh <bucket> [expected-region]

set -uo pipefail

BUCKET="${1:?usage: s3_check.sh <bucket> [expected-region]}"
EXPECT="${2:-}"

loc=$(aws s3api get-bucket-location --bucket "$BUCKET" --output text 2>&1)
rc=$?

if [ $rc -ne 0 ]; then
  echo "FAIL  cannot read s3://${BUCKET}"
  echo "${loc}" | sed 's/^/      /'
  case "$loc" in
    *"Token has expired"*|*"expired"*)
      echo "      Refresh SSO:  aws sso login --profile \${AWS_PROFILE:-<profile>}" ;;
    *"Unable to locate credentials"*)
      echo "      Set a profile:  export AWS_PROFILE=<profile>" ;;
    *NoSuchBucket*)
      echo "      The bucket does not exist, or belongs to another account." ;;
  esac
  exit 1
fi

# get-bucket-location returns "None" for us-east-1, which is a historical quirk.
[ "$loc" = "None" ] && loc="us-east-1"

echo "bucket s3://${BUCKET} is in ${loc}"

if [ -n "$EXPECT" ] && [ "$EXPECT" != "$loc" ]; then
  cat <<EOF
FAIL  region mismatch.
      bucket is in : ${loc}
      you are using: ${EXPECT}

      Transfers would cross regions: slower, and billed per GB. Worse, the GPU
      instance must sit in the SAME region as the bucket for the staging step to
      be worth doing at all.

      Fix by using the bucket's region:
        make s3-push AWS_REGION=${loc}

      Note AWS_REGION may be coming from your environment or your AWS profile
      rather than the Makefile. Check with:
        echo "\$AWS_REGION \$AWS_DEFAULT_REGION"
        aws configure get region
EOF
  exit 1
fi

exit 0
