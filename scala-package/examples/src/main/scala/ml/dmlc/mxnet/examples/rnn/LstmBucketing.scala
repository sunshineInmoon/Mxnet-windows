package ml.dmlc.mxnet.examples.rnn

import ml.dmlc.mxnet.Callback.Speedometer
import ml.dmlc.mxnet._
import ml.dmlc.mxnet.examples.rnn.BucketIo.BucketSentenceIter
import ml.dmlc.mxnet.optimizer.SGD
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/**
 * Bucketing LSTM examples
 * @author Yizhi Liu
 */
class LstmBucketing {
  @Option(name = "--data-train", usage = "training set")
  private val dataTrain: String = "example/rnn/ptb.train.txt"
  @Option(name = "--data-val", usage = "validation set")
  private val dataVal: String = "example/rnn/ptb.valid.txt"
  @Option(name = "--num-epoch", usage = "the number of training epoch")
  private val numEpoch: Int = 5
  @Option(name = "--gpus", usage = "the gpus will be used, e.g. '0,1,2,3'")
  private val gpus: String = null
  @Option(name = "--cpus", usage = "the cpus will be used, e.g. '0,1,2,3'")
  private val cpus: String = null
  @Option(name = "--save-model-path", usage = "the model saving path")
  private val saveModelPath: String = "model/lstm"
}

object LstmBucketing {
  private val logger: Logger = LoggerFactory.getLogger(classOf[LstmBucketing])

  def perplexity(label: NDArray, pred: NDArray): Float = {
    val batchSize = label.shape(0)
    // TODO: NDArray transpose
    val labelArr = Array.fill(label.size)(0)
    (0 until batchSize).foreach(row => {
      val labelRow = label.slice(row)
      labelRow.toArray.zipWithIndex.foreach { case (l, col) =>
        labelArr(col * batchSize + row) = l.toInt
      }
    })
    var loss = .0
    (0 until pred.shape(0)).foreach(i =>
      loss -= Math.log(Math.max(1e-10f, pred.slice(i).toArray(labelArr(i))))
    )
    Math.exp(loss / labelArr.length).toFloat
  }

  def main(args: Array[String]): Unit = {
    val inst = new LstmBucketing
    val parser: CmdLineParser = new CmdLineParser(inst)
    try {
      parser.parseArgument(args.toList.asJava)
      val contexts =
        if (inst.gpus != null) inst.gpus.split(',').map(id => Context.gpu(id.trim.toInt))
        else if (inst.cpus != null) inst.cpus.split(',').map(id => Context.cpu(id.trim.toInt))
        else Array(Context.cpu(0))

      val batchSize = 32
      val buckets = Array(10, 20, 30, 40, 50, 60)
      val numHidden = 200
      val numEmbed = 200
      val numLstmLayer = 2

      val learningRate = 0.01f
      val momentum = 0.0f

      logger.info("Building vocab ...")
      val vocab = BucketIo.defaultBuildVocab(inst.dataTrain)

      class BucketSymGen extends SymbolGenerator {
        override def generate(key: AnyRef): Symbol = {
          val seqLen = key.asInstanceOf[Int]
          Lstm.lstmUnroll(numLstmLayer, seqLen, vocab.size,
            numHidden = numHidden, numEmbed = numEmbed, numLabel = vocab.size)
        }
      }

      val initC = (0 until numLstmLayer).map(l =>
        (s"l${l}_init_c", (batchSize, numHidden))
      )
      val initH = (0 until numLstmLayer).map(l =>
        (s"l${l}_init_h", (batchSize, numHidden))
      )
      val initStates = initC ++ initH

      val dataTrain = new BucketSentenceIter(inst.dataTrain, vocab,
        buckets, batchSize, initStates)
      val dataVal = new BucketSentenceIter(inst.dataVal, vocab,
        buckets, batchSize, initStates)

      logger.info("Start training ...")
      val model = FeedForward.newBuilder(new BucketSymGen())
        .setContext(contexts)
        .setNumEpoch(inst.numEpoch)
        .setOptimizer(new SGD(learningRate = learningRate, momentum = momentum, wd = 0.00001f))
        .setInitializer(new Xavier(factorType = "in", magnitude = 2.34f))
        .setTrainData(dataTrain)
        .setEvalData(dataVal)
        .setEvalMetric(new CustomMetric(perplexity, name = "perplexity"))
        .setBatchEndCallback(new Speedometer(batchSize, 50))
        .build()
      model.save(inst.saveModelPath)
    } catch {
      case ex: Exception =>
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
    }
  }
}
