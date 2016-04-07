name := "naive-ml"

organization := "tupol"

version := "0.1.0"

scalaVersion := "2.10.4"


// ------------------------------
// DEPENDENCIES AND RESOLVERS

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "org.scalanlp" %% "breeze" % "0.12"
)

// ------------------------------
// RUNNING

// Make sure that provided dependencies are added to classpath when running in sbt
run in Compile <<= Defaults.runTask(fullClasspath in Compile,
  mainClass in(Compile, run),
  runner in(Compile, run))

fork in run := true
