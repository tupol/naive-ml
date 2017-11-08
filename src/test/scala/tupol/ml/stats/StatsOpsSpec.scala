package tupol.ml.stats

import org.scalatest.{FunSuite, Matchers}

import statsops._

class StatsOpsSpec extends FunSuite with Matchers {

  test("zero |+| zero") {
    Stats.zeroDouble |+| Stats.zeroDouble shouldBe Stats.zeroDouble
    Stats.zeroPoint |+| Stats.zeroPoint shouldBe Stats.zeroPoint
  }

  test("stats |+| zero") {
    Stats.fromDouble(1) |+| Stats.zeroDouble shouldBe Stats.fromDouble(1)
    val stats1 = Stats.fromPoint(Array(1.0))
    stats1 |+| Stats.zeroPoint shouldEqual stats1
  }

  test("stats1 |+| stats2 = stats2 |+| stats1") {
    Stats.fromDouble(1) |+| Stats.fromDouble(2) shouldBe Stats.fromDouble(2) |+| Stats.fromDouble(1)
    Stats.fromPoint(Array(1.0)) |+| Stats.fromPoint(Array(2.0)) shouldEqual (Stats.fromPoint(Array(2.0)) |+| Stats.fromPoint(Array(1.0)))
  }

}
