name := "naive-ml"

organization := "tupol"

version := "0.1.0"

scalaVersion := "2.11.7"


// ------------------------------
// DEPENDENCIES AND RESOLVERS

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.4" % "it,test",
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12",
  "org.scalanlp" %% "breeze-viz" % "0.12"
)


// ------------------------------
// TESTING

// INTEGRATION TESTS

lazy val integrationTestsRoot: Project = (project in file("."))
  .configs(IntegrationTest)
  .settings( Defaults.itSettings : _*)

testOptions in IntegrationTest += Tests.Filter(t => t endsWith "Test")

parallelExecution in IntegrationTest := false

fork in IntegrationTest := true

// ------------------------------
// RUNNING

// Make sure that provided dependencies are added to classpath when running in sbt
run in Compile <<= Defaults.runTask(fullClasspath in Compile,
  mainClass in(Compile, run),
  runner in(Compile, run))

fork in run := true

// ------------------------------

