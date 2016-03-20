name := "sparx-ml"

organization := "tupol"

version := "0.1.0"

scalaVersion := "2.10.4"

val sparkVersion = "1.6.0"

// ------------------------------
// DEPENDENCIES AND RESOLVERS

lazy val providedDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion force(),
  "org.apache.spark" %% "spark-sql" % sparkVersion force(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion force(),
  "org.apache.spark" %% "spark-streaming" % sparkVersion force()

)

libraryDependencies ++= providedDependencies
//  .map(_ % "provided")

libraryDependencies ++= Seq(
  "org.elasticsearch" %% "elasticsearch-spark" % "2.2.0" intransitive(),
  "org.apache.spark" %% "spark-streaming-kafka" % sparkVersion,
  "spark.jobserver" %% "job-server-api" % "0.6.1",
  "org.scalatest" %% "scalatest" % "2.2.4" % "test"
)

// ------------------------------
// RUNNING

// Make sure that provided dependencies are added to classpath when running in sbt
run in Compile <<= Defaults.runTask(fullClasspath in Compile,
  mainClass in(Compile, run),
  runner in(Compile, run))

fork in run := true

// ------------------------------
// ASSEMBLY
assemblyJarName in assembly := s"${name.value}-fat.jar"

// Exclude thee unmanaged library jars from local-lib
assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  cp filter {_.data.absolutePath.startsWith(unmanagedBase.value.getAbsolutePath)}
}

// Add exclusions, provided...
assemblyMergeStrategy in assembly := {
  {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => MergeStrategy.first
  }
}

artifact in (Compile, assembly) := {
  val art = (artifact in (Compile, assembly)).value
  art.copy(`classifier` = Some("assembly"))
}

addArtifact(artifact in (Compile, assembly), assembly)

// ------------------------------
