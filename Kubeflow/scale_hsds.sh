# change the number of HSDS pods
# Usage: ./scale_hsds.sh <pod_count>
kubectl scale --replicas=$1 deployment/hsds
