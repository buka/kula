java -Djava.library.path=./lib -Dfile.encoding=UTF8 -Xmx1536M -Xss1M -XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=512M  -jar `dirname $0`/sbt-launch.jar "$@"
