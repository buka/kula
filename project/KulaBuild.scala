import sbt._
import Keys._

object KulaBuild extends Build {
  import Resolvers._

  lazy val buildSettings = Seq(
    organization := "buka",
    version      := "0.1-SNAPSHOT",
    scalaVersion := "2.9.0-1"
  )

  lazy val kula = Project(
    id = "kula",
    base = file("."),
    settings = defaultSettings ++ Seq(
      libraryDependencies ++= Dependencies.kula
    )
  )

  // Settings

  override lazy val settings = super.settings ++ buildSettings //++ Publish.versionSettings

  lazy val baseSettings = Defaults.defaultSettings //++ Publish.settings

  lazy val defaultSettings = baseSettings ++ Seq(
    resolvers ++= Seq(akkaSnapshots),

    // compile options
    scalacOptions ++= Seq("-encoding", "UTF-8", "-optimise", "-deprecation", "-unchecked"),
    javacOptions  ++= Seq("-Xlint:unchecked", "-Xlint:deprecation"),

    // add config dir to classpaths
    //unmanagedClasspath in Runtime <+= (baseDirectory in LocalProject("kula")) map { base => Attributed.blank(base / "config") },
    //unmanagedClasspath in Test    <+= (baseDirectory in LocalProject("kula")) map { base => Attributed.blank(base / "config") },

    // disable parallel tests
    parallelExecution in Test := false
  )
}

object Resolvers
{
  val akkaSnapshots = "Typesafe Snapshots Repo" at "http://repo.typesafe.com/typesafe/snapshots"
}

object Dependencies 
{
  import Dependency._

  val kula = Seq(akka_actor, logback)
}

object Dependency
{
  object Versions 
  {
    val Akka      = "2.0-20110716-000330"
    val Logback   = "0.9.28"
  }

  val akka_actor    = "se.scalablesolutions.akka"   % "akka-actor"      % Versions.Akka
  val logback       = "ch.qos.logback"      % "logback-classic" % Versions.Logback
}

