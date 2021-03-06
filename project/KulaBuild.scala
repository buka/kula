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

  lazy val kula_samples = Project(
    id = "kula-samples",
    base = file("samples"),
    settings = defaultSettings ++ Seq(
      libraryDependencies ++= Dependencies.kula_samples
    )
  ) dependsOn(kula)

  // Settings

  override lazy val settings = super.settings ++ buildSettings //++ Publish.versionSettings

  lazy val baseSettings = Defaults.defaultSettings //++ Publish.settings

  def specs2Framework = new TestFramework("org.specs2.runner.SpecsFramework")
  lazy val defaultSettings = baseSettings ++ Seq(
    resolvers ++= Seq(scalaToolsReleases, scalaToolsSnapshots, akkaSnapshots, sbtIdea),

    // compile options
    scalacOptions ++= Seq("-encoding", "UTF-8", "-optimise", "-deprecation", "-unchecked"),
    javacOptions  ++= Seq("-Xlint:unchecked", "-Xlint:deprecation"),

    // add config dir to classpaths
    //unmanagedClasspath in Runtime <+= (baseDirectory in LocalProject("kula")) map { base => Attributed.blank(base / "config") },
    //unmanagedClasspath in Test    <+= (baseDirectory in LocalProject("kula")) map { base => Attributed.blank(base / "config") },

    // disable parallel tests
    parallelExecution in Test := false,
    testFrameworks := Seq(specs2Framework) 
  )
}

object Resolvers
{
  val akkaSnapshots       = "Typesafe Snapshots Repo" at "http://repo.typesafe.com/typesafe/snapshots"
  val scalaToolsSnapshots = "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots"
  val scalaToolsReleases  = "Scala Tools Releases" at "http://scala-tools.org/repo-releases"
  val sbtIdea             = "SBT IDEA Repo" at "http://mpeltonen.github.com/maven/"
}

object Dependencies 
{
  import Dependency._

  val kula          = Seq(akka_actor, commons_codec, Test.specs2, Test.logback)
  val kula_samples  = Seq(akka_actor)

}

object Dependency
{
  object Versions 
  {
    val Akka          = "2.0-20110716-000330"
    val Logback       = "0.9.28"
    val Specs2        = "1.5"
    val CommonsCodec  = "1.5"
  }

  val akka_actor    = "se.scalablesolutions.akka"   %   "akka-actor"      % Versions.Akka
  val commons_codec = "commons-codec"               % "commons-codec"     % Versions.CommonsCodec

  object Test
  {
    val logback     = "ch.qos.logback"              % "logback-classic" % Versions.Logback    % "test"
    val specs2      = "org.specs2"                  %% "specs2"         % Versions.Specs2     % "test"
  }
}

