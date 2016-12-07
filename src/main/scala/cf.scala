import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

object cf {

	  def Str2Int(s : String) :Int = {
		val num = s match{
			case "Action" => 0
			case "Adventure" => 1
			case "Animation" => 2
			case "Children's" => 3
			case "Comedy" => 4
			case "Crime" => 5
			case "Documentary" => 6
			case "Drama" => 7
			case "Fantasy" => 8
			case "Film-Noir" => 9
			case "Horror" => 10
			case "Musical" => 11
			case "Mystery" => 12
			case "Romance" => 13
			case "Sci-Fi" => 14
			case "Thriller" => 15
			case "War" => 16
			case "Western" => 17
		}
		num
	}

	  def main(args: Array[String]) = {
	  		val conf = new SparkConf().setAppName("CF")
	  		val sc = new SparkContext(conf)
	  		// Load and parse the data
			val data = sc.textFile("ml-1m/ratings.dat")
			val MovieData = sc.textFile("ml-1m/movies.dat")
			val Originalratings = data.map(_.split("::").take(3) match { case Array(user, item, rate) =>
			  Rating(user.toInt, item.toInt, rate.toDouble)
			})

			val (train, test) = Originalratings.randomSplit(Array(0.9, 0.1)) match {
				case Array(train, test) => (train, test) 
			}	

			//val MovieRating = Originalratings.map{case Rating(user, item, rate) => (item.toInt, (user.toInt, rate.toDouble))}
			val MovieRating = train.map{case Rating(user, item, rate) => (item.toInt, (user.toInt, rate.toDouble))}
			val MovieTag = MovieData.map(_.split("::").take(3) match {
				case Array(movie, title, tag) => (movie.toInt, tag) 
			}).flatMapValues(x => x.split('|'))

			val temp = MovieRating.join(MovieTag).map{case (item, ((user, rate), tag)) => ((user.toInt,tag), rate.toDouble) }

			val avgValue = temp.mapValues(x => (x,1)).reduceByKey((x,y) => (x._1+y._1,x._2+y._2)).mapValues{case (sum, count) => (sum/count)}
			
			val ratings = avgValue.map{case ((user, tag), rate) => Rating(user,Str2Int(tag), rate)}

			// val (train, test) = ratings.randomSplit(Array(0.9, 0.1)) match {
			// 	case Array(train, test) => (train, test) 
			// }			
			// Build the recommendation model using ALS
			val rank = 10
			val numIterations = 10
			// val model = ALS.train(train, rank, numIterations, 0.01)
			val model = ALS.train(ratings, rank, numIterations, 0.01)

			// Evaluate the model on rating data
			// val usersProducts = ratings.map { case Rating(user, product, rate) =>
			//   (user, product)
			// } 

			// //count the number of ratings given by each user format:(user, count) and filter
			// val userCount = Originalratings.map{case Rating(user, movie, rate) => (user, 1)}.reduceByKey((x, y) => (x+y))
			// .filter{case (user, count) => count>500}

			// //join with the Originalratings
			// val fullRating = Originalratings.map{case Rating(user, movie, rate) => (user, (movie, rate))}
			// val specialTest = userCount.join(fullRating).map{case (user, (count, (movie, rate))) => Rating(user, movie, rate)}



			//count the number of ratings for each movie format:(movie, count) and filter
			val movieCount = test.map{case Rating(user, movie, rate) => (movie, 1)}.reduceByKey((x, y) => (x+y))
			.filter{case (movie, count) => count < 30}

			//join with the Originalratings format:(movie, (count, (user, rate)))
			val fullRating = test.map{case Rating(user, movie, rate) => (movie, (user, rate))}
			val specialTest = movieCount.join(fullRating).map{case (movie, (count, (user, rate))) => Rating(user, movie, rate)}

			// val (train1, test1) = Originalratings.randomSplit(Array(0.5, 0.5)) match {
			// 	case Array(redundant, test1) => (redundant, test1)
			// }
			val OriginalModel = ALS.train(train, rank, numIterations, 0.01)
			//val OriginalMSE = computeMSE(OriginalModel, test1)
			val OriginalMSE = computeMSE(OriginalModel, specialTest)

			//translate test data
			//val movieUser = test1.map{case Rating(user, movie, rate) => (movie, user)}
			val movieUser = specialTest.map{case Rating(user, movie, rate) => (movie, user)}

			//create movieTagInt
			val movieTagInt = MovieTag.map{case (movie, tag) => (movie, Str2Int(tag))}
			//create (movie,(user,tag))
			val MUT = movieUser.join(movieTagInt)

			//create UMP((user,movie), prediction) from user-tag model
			val UT = MUT.map{case (movie, (user, tag)) => (user, tag)}
			val P = model.predict(UT).map{
				case Rating(user, tag, rate) => ((user, tag), rate) 
			}

			val UMP = MUT.map{case (movie, (user, tag)) => ((user, tag), movie)}.join(P)
			.map{case ((user, tag), (movie, rate)) => ((user, movie), rate)}
			//Reduce UMP by Key
			val UMP2 = UMP.mapValues(x => (x, 1)).reduceByKey((x, y) => (x._1+y._1, x._2+y._2)).mapValues{case (sum, count) => (sum/count)}
			//val testMSE = computeMSE(model, test)
			//val ratesAndPreds = test1.map{case Rating(user, movie, rate) => ((user, movie), rate)}.join(UMP2)
			val ratesAndPreds = specialTest.map{case Rating(user, movie, rate) => ((user, movie), rate)}.join(UMP2)
			val MSE = ratesAndPreds.map{case ((user, movie), (r1, r2)) => 
				val err = (r1-r2)
				err*err
			}.mean()
			println("*********************************************************")
			println(s"MSE = $MSE")
			println(s"OriginalMSE = $OriginalMSE")
			println("*********************************************************")
			// model.save(sc, "target/tmp/myCollaborativeFilter")
			// val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
	  }

	  def computeMSE(model: MatrixFactorizationModel, data: RDD[Rating]) = {
	    val usersProducts = data.map { case Rating(user, product, rate) =>
	      (user, product)
	    }
	    val predictions = model.predict(usersProducts).map {
	      case Rating(user, product, rate) => ((user, product), rate)
	    }
	    val ratesAndPreds = data.map { case Rating(user, product, rate) =>
	      ((user, product), rate)
	    }.join(predictions)
	    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
	      val err = (r1 - r2)
	      err * err
	    }.mean()
	    MSE
	  }
	
}
