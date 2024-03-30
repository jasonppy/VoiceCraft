:: Windows Docker Context Batch File
:: No need to install nvidia toolkit as Windows driver supports docker GPUs
:: Credit to github/jay-c88 for this
:: https://github.com/jasonppy/VoiceCraft/pull/25#issuecomment-2028053878
@echo off

docker start jupyter > nul 2> nul || ^
docker run -it ^
-d ^
--gpus all ^
-p 8888:8888 ^
--name jupyter ^
--user root ^
-e NB_USER="%username%" ^
-e CHOWN_HOME=yes ^
-e GRANT_SUDO=yes ^
-w "/home/%username%" ^
-v %cd%:"/home/%username%/work" ^
jupyter/base-notebook

pause
