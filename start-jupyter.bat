@echo off

echo Creating and running the Jupyter container...

docker run -it -d ^
    --gpus all ^
    -p 8888:8888 ^
    -p 7860:7860 ^
    --name jupyter ^
    --user root ^
    -e NB_USER="%username%" ^
    -e CHOWN_HOME=yes ^
    -e GRANT_SUDO=yes ^
    -e JUPYTER_TOKEN=mytoken ^
    -w "/home/%username%" ^
    -v "%cd%":"/home/%username%/work" ^
    voicecraft

if %errorlevel% == 0 (
    echo Jupyter container created and running.

    echo Jupyter container is running.
    echo To access the Jupyter web UI, please follow these steps:
    echo 1. Open your web browser
    echo 2. Navigate to http://localhost:8888/?token=mytoken
    echo 3. !! The default token is "mytoken" and should be changed. !!
    pause
) else (
    echo Failed to create and run the Jupyter container.
)
