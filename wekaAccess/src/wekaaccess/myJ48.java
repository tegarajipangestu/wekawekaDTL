package wekaaccess;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
 * @author tegar
 */
public class myJ48
        extends Classifier {

    Instances instances;
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
     * Index of class value if node is leaf.
     */
    public int leaf_class_idx;

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

    public int maxAttr(Instances data, Attribute atr) throws Exception {
        System.out.println(atr.toString());
        int[] maxval = new int[atr.numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance temp = data.instance(i);
            maxval[(int) temp.classValue()]++;
        }
        return Utils.maxIndex(maxval);
    }

    private void handleMissingValue(Instances data) {

        Enumeration attrEnum = data.enumerateAttributes();
        while (attrEnum.hasMoreElements()) {
            Attribute attr = (Attribute) attrEnum.nextElement();
            //Handling nominal, just assign it with majority value
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
        instances = data;
        if (data.numInstances() == 0) {
            split_attribute = null;
            leaf_class_idx = -1;
            leaf_distribution = new double[data.numClasses()];
            return;
        }

        // Compute attribute with maximum gain ratio
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
        }
        // Make leaf if gain ratio is zero. 
        // Otherwise create successors.
        if (Utils.eq(gainRatio[Utils.maxIndex(gainRatio)], 0)) {
            split_attribute = null;
            leaf_distribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                leaf_distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(leaf_distribution);
            leaf_class_idx = Utils.maxIndex(leaf_distribution);
            class_attribute = data.classAttribute();
        } else {
            split_attribute = data.attribute(Utils.maxIndex(gainRatio));
            Instances[] splitData;
            int numChild;
            if (split_attribute.isNominal()) {
                numChild = split_attribute.numValues();
                splitData = splitData(data, split_attribute);
            } else {
                numChild = 2;
                splitData = splitData(data, split_attribute, getOptimumThreshold(data, split_attribute));
            }
            child = new myJ48[numChild];
            for (int j = 0; j < numChild; j++) {
                child[j] = new myJ48();
                child[j].makeTree(splitData[j]);
                if (Utils.eq(splitData[j].numInstances(), 0)) {
                    child[j].leaf_class_idx = maxAttr(data, data.classAttribute());
                }
            }

            for (int i = 0; i < numChild; i++) {
                if (child[i].leaf_class_idx != 0 && Utils.eq(child[i].leaf_class_idx, -999)) {
                    double[] classDistribution = new double[data.numClasses()];
                    Enumeration instanceEnum = data.enumerateInstances();
                    while (instanceEnum.hasMoreElements()) {
                        Instance instance = (Instance) instanceEnum.nextElement();
                        classDistribution[(int) instance.classValue()]++;
                    }
                    Utils.normalize(classDistribution);
                    child[i].leaf_class_idx = Utils.maxIndex(classDistribution);
                    child[i].leaf_distribution = classDistribution;
                }
            }
            pruneTree();
        }
    }

    @Override
    public double classifyInstance(Instance instance)
            throws Exception {

        if (instance.hasMissingValue()) {
            throw new Exception("This will never happens, sure");
        }
        if (split_attribute == null) {
            {
                if (!Utils.eq(leaf_class_idx, Double.NaN)) {
                    return leaf_class_idx;
                } else {
                    Enumeration a = instance.enumerateAttributes();
                    return instance.value(class_attribute);
                }
            }
        } else {
            if (split_attribute.isNumeric()) {
                int numericAttrIdx = -1;
                if (instance.value(split_attribute) > getOptimumThreshold(instances, split_attribute)) {
                    numericAttrIdx = 1;
                } else {
                    numericAttrIdx = 0;
                }
                return child[(int) numericAttrIdx].
                        classifyInstance(instance);
            } else if (split_attribute.isNominal()) {
                return child[(int) instance.value(split_attribute)].
                        classifyInstance(instance);
            } else {
                throw new Exception("This will never happens, sure");
            }
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance)
            throws Exception {
        if (split_attribute != null) {
            double split_attribute_idx = 0;
            if (split_attribute.isNominal()) {
                split_attribute_idx = instance.value(split_attribute);
                if (Double.isNaN(split_attribute_idx)) {
                    Instances[] instancesSplitted = splitData(instances, split_attribute);
                    int largestNumIdx = -1;
                    int cnt = 0;
                    for (int i = 0; i < instancesSplitted.length; ++i) {
                        int tmp = instancesSplitted[i].numInstances();
                        if (tmp > cnt) {
                            largestNumIdx = i;
                        }
                    }
                    split_attribute_idx = largestNumIdx;
                }
                if (split_attribute_idx == -1) {
                    throw new Exception("This will never happens, sure");
                }
            } else if (split_attribute.isNumeric()) {
                double val = instance.value(split_attribute);
                if (Double.isNaN(val)) {
                    throw new Exception("This will never happens, sure");
                } else {
                    //manual classifying
                    if (val >= getOptimumThreshold(instances, split_attribute)) {
                        split_attribute_idx = 1;
                    } else {
                        split_attribute_idx = 0;
                    }
                }
            }
            if (child.length > 0) {
                return child[(int) split_attribute_idx].distributionForInstance(instance);
            }
            if (leaf_distribution != null) {
                return leaf_distribution;
            } else {
                System.out.println("Halo sayang");
            }
        } else {
            return leaf_distribution;
        }
        if (leaf_distribution != null) {
            return leaf_distribution;
        } else {
            return null;
        }
    }

    public double computeGainRatio(Instances data, Attribute attr) throws Exception {

        double infoGain = 0.0;
        Instances[] splitData = myJ48.this.splitData(data, attr);
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
        Instances[] splitData = splitData(data, attr, threshold);
        infoGain = computeEntropy(data);
        for (int i = 0; i < 2; i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain = infoGain - (double) splitData[i].numInstances()
                        / (double) data.numInstances() * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

    public Instances[] splitData(Instances data, Attribute attr, double threshold) throws Exception {
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
            splitedData[i] = new Instances(data, data.numInstances());
        }

        Enumeration instanceIterator = data.enumerateInstances();
        while (instanceIterator.hasMoreElements()) {
            Instance instance = (Instance) instanceIterator.nextElement();
            splitedData[(int) instance.value(attr)].add(instance);
        }

        for (Instances instances : splitedData) {
            instances.compactify(); //WEKA said it so, for the sake of optimizing
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

    public double computeError(Instances instances) throws Exception {
        int correctInstances = 0;
        int incorrectInstances = 0;
        Enumeration enumeration = instances.enumerateInstances();
        while (enumeration.hasMoreElements()) {
            Instance instance = (Instance) enumeration.nextElement();
            if (instance.classValue() == classifyInstance(instance)) {
                correctInstances++;
            } else {
                incorrectInstances++;
            }
        }
        return (double) incorrectInstances / (double) (incorrectInstances + correctInstances);
    }

    private void pruneTree() throws Exception {
        //Prepruning, prune before its too late, beybeh
        if (child != null) {
            double beforePruningError = this.computeError(instances);

            double[] classDistribution = new double[instances.numClasses()];
            Enumeration instanceEnum = instances.enumerateInstances();
            while (instanceEnum.hasMoreElements()) {
                Instance instance = (Instance) instanceEnum.nextElement();
                classDistribution[(int) instance.classValue()]++;
            }
            Utils.normalize(classDistribution);
            int idxClass = Utils.maxIndex(classDistribution);

            int correctInstances = 0;
            int incorrectInstances = 0;
            Enumeration enumeration = instances.enumerateInstances();
            while (enumeration.hasMoreElements()) {
                Instance instance = (Instance) enumeration.nextElement();
                if (instance.classValue() == classifyInstance(instance)) {
                    correctInstances++;
                } else {
                    incorrectInstances++;
                }
            }
            double afterPruningError = (double) incorrectInstances / (double) (correctInstances + incorrectInstances);
            if (beforePruningError > afterPruningError) {
                System.out.println("Pruning, behold the power");
                child = null;
                split_attribute = null;
                leaf_class_idx = idxClass;
                leaf_distribution = classDistribution;
            }

        }

    }

}
