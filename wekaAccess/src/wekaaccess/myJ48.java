/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaaccess;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;
import weka.core.AttributeStats;

/**
 *
 * @author Ivana Clairine
 */
public class myJ48
        extends Classifier {

    /**
     * for serialization
     */
    static final long serialVersionUID = -2693678647096322561L;

    /**
     * The node's successors.
     */
    private myJ48[] child;

    /**
     * Attribute used for splitting.
     */
    private Attribute split_attribute;

    /**
     * Class value if node is leaf.
     */
    private double leaf_class;

    /**
     * Class distribution if node is leaf.
     */
    private double[] leaf_distribution;

    /**
     * Class attribute of dataset.
     */
    private Attribute class_attribute;

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // handling missing value
        handleMissingValue(data);

        makeTree(data);
    }

    public int maxAttr(Instances data, Attribute atr) {
        Instances[] maxAttr = splitData(data, atr);
        int[] maxval = new int[atr.numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance temp = data.instance(i);
            maxval[(int) temp.classValue()]++;
        }
        return findmax(maxval);
    }

    public int findmax(int[] input) {
        int max = -1;
        for (int counter = 1; counter < input.length; counter++) {
            if (input[counter] > max) {
                max = counter;
            }
        }
        return max;
    }

    private void handleMissingValue(Instances data) {

        Enumeration attrEnum = data.enumerateAttributes();
        while (attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            //Handling nominal, just assign it with majority class
            if (attr.isNominal()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                int maxIndex = 0;
                for (int i = 1; i < attr.numValues(); i++) {
                    if (attributeStats.nominalCounts[maxIndex] < attributeStats.nominalCounts[i]) {
                        maxIndex = i;
                    }
                }
                Enumeration instEnum = data.enumerateInstances();
                while (instEnum.hasMoreElements()) {
                    Instance instance = (Instance) instEnum.nextElement();
                    if (instance.isMissing(attr.index())) {
                        instance.setValue(attr.index(), maxIndex);
                    }
                }
            } //Handling numeric, just assign it with mean of attribute's instances
            else if (attr.isNumeric()) {
                AttributeStats attributeStats = data.attributeStats(attr.index());
                double mean = attributeStats.numericStats.mean;
                if (Double.isNaN(mean)) {
                    mean = 0;
                }
                Enumeration instEnumerate = data.enumerateInstances();
                while (instEnumerate.hasMoreElements()) {
                    Instance instance = (Instance) instEnumerate.nextElement();
                    if (instance.isMissing(attr.index())) {
                        instance.setValue(attr.index(), mean);
                    }
                }
            }
        }
    }

    private void makeTree(Instances data) throws Exception {
        // Check if no instances have reached this node.
        if (data.numInstances() == 0) {
            split_attribute = null;
            leaf_class = Double.NaN;
            leaf_distribution = new double[data.numClasses()];
            return;
        }

        // Compute attribute with maximum gain ratio.
        double[] gainRatio = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            if (att.isNominal()) {
                //kasus normal
                gainRatio[att.index()] = computeGainRatio(data, att);
            } else if (att.isNumeric()) {
                //kasus tidak normal
                gainRatio[att.index()] = computeGainRatio(data, att, getOptimumThreshold(data, att));
            }

            gainRatio[att.index()] = computeGainRatio(data, att) / hitungEntropy(data);
        }
        split_attribute = data.attribute(Utils.maxIndex(gainRatio));

        // Make leaf if gain ratio is zero. 
        // Otherwise create successors.
        if (Utils.eq(gainRatio[split_attribute.index()], 0)) {
            split_attribute = null;
            leaf_distribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                leaf_distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(leaf_distribution);
            leaf_class = Utils.maxIndex(leaf_distribution);
            //leaf_class = maxAttr(data, split_attribute);
            class_attribute = data.classAttribute();
        } else {
            Instances[] splitData = splitData(data, split_attribute);
            child = new myJ48[split_attribute.numValues()];
            for (int j = 0; j < split_attribute.numValues(); j++) {
                child[j] = new myJ48();
                child[j].makeTree(splitData[j]);
                if (Utils.eq(splitData[j].numInstances(), 0)) {
                    child[j].leaf_class = maxAttr(data, data.attribute(j));
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance)
            throws Exception {

        if (instance.hasMissingValue()) {
            throw new Exception("Can't handle missing value(s)");
        }
        if (split_attribute == null) {
            {
                if (!Utils.eq(leaf_class, Double.NaN)) {
                    return leaf_class;
                } else {
                    //return instance.classAttribute().;
                    Enumeration a = instance.enumerateAttributes();
                    return instance.value(class_attribute);
                }
            }
        } else {
            return child[(int) instance.value(split_attribute)].
                    classifyInstance(instance);
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance)
            throws Exception {
        if (instance.hasMissingValue()) {
            throw new Exception("Can't handle missing value(s)");
        }
        if (split_attribute == null) {
            return leaf_distribution;
        } else {
            return child[(int) instance.value(split_attribute)].
                    distributionForInstance(instance);
        }
    }

    private double hitungEntropy(Instances data) {
        double[] kelas = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance temp = data.instance(i);
            kelas[(int) temp.classValue()]++;
        }
        for (int i = 0; i < data.numClasses(); i++) {
            kelas[i] = kelas[i] / data.numInstances();
        }
        double entropi = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            if (kelas[i] > 0) {
                entropi = entropi - (kelas[i] * Utils.log2(kelas[i]));
            }
        }
        return entropi;
    }

    public double computeGainRatio(Instances data, Attribute attr) throws Exception {

        double infoGain = 0.0;
        Instances[] splitData = splitData(data, attr);
        infoGain = computeEntropy(data);
        for (int i = 0; i < attr.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= (double) splitData[i].numInstances()
                        / (double) data.numInstances() * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public double computeGainRatio(Instances data, Attribute attr, double threshold) throws Exception {

        double infoGain = 0.0;
        Instances[] splitData = splitNumericAttr(data, attr, threshold);
        infoGain = computeEntropy(data);
        for (int i = 0; i < 2; i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= (double) splitData[i].numInstances()
                        / (double) data.numInstances() * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public Instances[] splitNumericAttr(Instances data, Attribute attr, double threshold) throws Exception {
        Instances[] splitedData = new Instances[2];
        for (int i = 0; i < 2; i++) {
            splitedData[i] = new Instances(data, data.numInstances()); // initialize with data template and max capacity
        }

        Enumeration instanceIterator = data.enumerateInstances();
        while (instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            if (instance.value(attr) >= threshold) {
                splitedData[1].add(instance);
            } else {
                splitedData[0].add(instance);
            }
        }

        for (Instances instances : splitedData) {
            instances.compactify(); //WEKA said it so
        }

        return splitedData;
    }

    public Instances[] splitData(Instances data, Attribute attr) {

        Instances[] splitedData = new Instances[attr.numValues()];
        for (int i = 0; i < attr.numValues(); i++) {
            splitedData[i] = new Instances(data, data.numInstances()); // initialize with data template and max capacity
        }

        Enumeration instanceIterator = data.enumerateInstances();
        while (instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            splitedData[(int) instance.value(attr)].add(instance);
        }

        for (Instances instances : splitedData) {
            instances.compactify(); //WEKA said it so
        }

        return splitedData;
    }

    public double computeEntropy(Instances data) {
        // This fucking validation is a must
        if (data.numInstances() == 0) {
            return 0.0;
        }

        double[] classCounts = new double[data.numClasses()];
        Enumeration instanceIterator = data.enumerateInstances();
        int totalInstance = 0;
        while (instanceIterator.hasMoreElements()) {
            Instance inst = (Instance) instanceIterator.nextElement();
            classCounts[(int) inst.classValue()]++;
            totalInstance++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            double fraction = classCounts[j] / totalInstance;
            if (fraction != 0) {
                entropy -= fraction * Utils.log2(fraction);
            }
        }

        return entropy;
    }

    private double getOptimumThreshold(Instances data, Attribute attribute) throws Exception {
        double[] threshold = new double[data.numInstances()];
        double[] gainRatio = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances() - 1; ++i) {
            if (data.instance(i).classValue() != data.instance(i + 1).classValue()) {
                threshold[i] = (data.instance(i).value(attribute) + data.instance(i + 1).value(attribute)) / 2;
                gainRatio[i] = computeGainRatio(data, attribute, threshold[i]);
            }
        }
        double result = (double) threshold[Utils.maxIndex(gainRatio)];
        return result;
    }

}
