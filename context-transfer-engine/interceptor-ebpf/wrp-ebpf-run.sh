#!/bin/bash
set -e

INTERCEPTOR="${INTERCEPTOR:-./bin/interceptor-user}"

# Find the '--' separator
APP_START=0
for i in "${!@}"; do
	if [[ "${!i}" == "--" ]]; then
		APP_START=$i
		break
	fi
done

if [[ $APP_START -eq 0 ]]; then
	echo "Usage: $0 [options] -- <application> [args...]"
	exit 1
fi

# Check interceptor exists
if [[ ! -x "$INTERCEPTOR" ]]; then
	echo "Error: interceptor-user not found at $INTERCEPTOR"
	echo "Set INTERCEPTOR env var or ensure binary exists"
	exit 1
fi

# Cleanup function
cleanup() {
	if [[ -n $INTERCEPTOR_PID ]] && kill -0 $INTERCEPTOR_PID 2>/dev/null; then
		sudo kill $INTERCEPTOR_PID 2>/dev/null || true
	fi
	exit 0
}
trap cleanup EXIT SIGINT SIGTERM

# Get app command (everything after --)
shift $APP_START
APP_CMD="$1"
shift
APP_ARGS="$@"

# Start the app
echo "Starting: $APP_CMD $APP_ARGS"
$APP_CMD $APP_ARGS &
APP_PID=$!
echo "App PID: $APP_PID"

# Start eBPF interceptor
echo "Attaching eBPF interceptor..."
sudo "$INTERCEPTOR" --root-pid "$APP_PID" &
INTERCEPTOR_PID=$!
echo "Interceptor PID: $INTERCEPTOR_PID"

# Wait for app to finish
wait $APP_PID
APP_EXIT=$?

echo "App exited with code $APP_EXIT"
exit $APP_EXIT
