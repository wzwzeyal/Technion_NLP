FROM_SCREEN=1
TO_SCREEN=1

for SCREEN_NUM in $(seq $FROM_SCREEN $TO_SCREEN)
do
  SCREEN_NAME="train${SCREEN_NUM}"

  # Create New Screens
  screen -S ${SCREEN_NAME} -d -m bash
  echo "Created ${SCREEN_NAME}"

  # Run some command
  screen -r ${SCREEN_NAME} -X stuff "echo hello"
  echo "Ran command in ${SCREEN_NAME}"
done