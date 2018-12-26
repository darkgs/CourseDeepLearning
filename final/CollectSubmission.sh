
echo "Make zip file...."
if [ -z "$1" ]; then
    echo "Team number is required.
Usage: ./CollectSubmission team_#"
    exit 0
fi

rm -f $1.tar.gz
mkdir $1
cp -r autostart_practice.sh autostart.sh driver_agent.py gym_torcs.py gym_torcs_test.py my_config.py reward_function.py testGame.py training.py snakeoil3_gym.py torcs_model $1/
tar cvzf $1.tar.gz $1

echo "Done."
