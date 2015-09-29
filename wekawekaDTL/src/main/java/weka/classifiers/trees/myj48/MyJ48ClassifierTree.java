/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.trees.myj48;

/**
 *
 * @author tegar
 */
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class MyJ48ClassifierTree {

    /**
     * Anak dari node
     */
    protected MyJ48ClassifierTree[] child;

    /**
     * Bapak dari node, you dont say
     */
    protected MyJ48ClassifierTree parent;

    /**
     * Bernilai true jika node ini adalah daun
     */
    protected boolean isLeaf;

    /**
     * Bernilai true jika node ini memiliki instance kosong
     */
    protected boolean isEmpty;

    /**
     * Training instance dari node ini
     */
    protected Instances trainInstances;

    /**
     * Pruning instances dari node ini
     */
    protected Distribution pruningInstances;

    /**
     * Id dari node ini
     */
    protected int idNode;

    protected String currentClass;

    /**
     * Constructor.
     */
    public MyJ48ClassifierTree() {

    }

    public void buildClassifier(Instances data) throws Exception {

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        buildTree(data, false);
    }

    public boolean isSingleClass(Instances data) {
        return data.numDistinctValues(data.classIndex()) == 1;
    }

    public boolean isEmptyInstances(Instances data) {
        return data.numInstances() == 0;
    }

    private double computeEntropy(Instances data) throws Exception {

        double[] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= (double) data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }

    private Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
    }

    private double computeInfoGain(Instances data, Attribute att)
            throws Exception {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances()
                        / (double) data.numInstances())
                        * computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    public Instances[] split(Instances data) throws Exception {

        Attribute m_Attribute;

        double m_ClassValue;

        double[] m_Distribution;

        Attribute m_ClassAttribute;

        m_Attribute = null;
        m_ClassValue = Instance.missingValue();
        m_Distribution = new double[data.numClasses()];
        double[] infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();

        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = computeInfoGain(data, att);
        }
        m_Attribute = data.attribute(Utils.maxIndex(infoGains));

        if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
            return null;
        } else {
            Instances[] splitData = splitData(data, m_Attribute);
            return splitData;
        }
    }

    public String getMajorityClassinParent() throws Exception {
        Instances instances = parent.trainInstances;

        return getMajorityClass(instances);
    }

    public String getMajorityClass(Instances _instances) throws Exception {
        int indexClass = _instances.classIndex();
        Instances instances = _instances;
        instances.sort(indexClass);
        List<String> className = new ArrayList<String>();
        for (int i = 0; i < instances.numInstances() - 1; i++) {
            int size = 0;
            if (instances.instance(i).value(indexClass) != instances.instance(i + 1).value(indexClass)) {
                className.add(instances.instance(i).stringValue(indexClass));
            }
        }

        return instances.instance(0).stringValue(indexClass);
    }

    public MyJ48ClassifierTree getParent() {
        return parent;
    }

    public Instances MyNumerictoNominal(Instances instances, int index) {
        Instances localInstances = instances; 
        localInstances.sort(index);
        if (localInstances.attribute(index).isNumeric())
        {
            
        }
        return null;
    }

    public final ClassifierSplitModel selectModel(Instances data) {

        double minResult;
        double currentResult;
        MyJ48Split[] currentModel;
        MyJ48Split bestModel = null;
        ClassifierSplitModel noSplitModel = null;
        double averageInfoGain = 0;
        int validModels = 0;
        boolean multiVal = true;
        Distribution checkDistribution;
        Attribute attribute;
        double sumOfWeights;
        int i;

        try {

            checkDistribution = new Distribution(data);
            int m_minNoObj = 0;
            if (Utils.sm(checkDistribution.total(), 2 * m_minNoObj)
                    || Utils.eq(checkDistribution.total(),
                            checkDistribution.perClass(checkDistribution.maxClass()))) {
                return noSplitModel;
            }

            if (data != null) {
                Enumeration enu = data.enumerateAttributes();
                while (enu.hasMoreElements()) {
                    attribute = (Attribute) enu.nextElement();
                    if ((attribute.isNumeric())
                            || (Utils.sm((double) attribute.numValues(),
                                    (0.3 * (double) data.numInstances())))) {
                        multiVal = false;
                        break;
                    }
                }
            }

            currentModel = new MyJ48Split[data.numAttributes()];
            sumOfWeights = data.sumOfWeights();

            for (i = 0; i < data.numAttributes(); i++) {

                if (i != (data).classIndex()) {

                    currentModel[i] = new MyJ48Split(i, data, sumOfWeights);
                    currentModel[i].buildClassifier(data);

                    if (currentModel[i].checkModel()) {
                        if (data != null) {
                            if ((data.attribute(i).isNumeric())
                                    || (multiVal || Utils.sm((double) data.attribute(i).numValues(),
                                            (0.3 * (double) data.numInstances())))) {
                                averageInfoGain = averageInfoGain + currentModel[i].infoGain();
                                validModels++;
                            }
                        } else {
                            averageInfoGain = averageInfoGain + currentModel[i].infoGain();
                            validModels++;
                        }
                    }
                } else {
                    currentModel[i] = null;
                }
            }

            if (validModels == 0) {
                return noSplitModel;
            }
            averageInfoGain = averageInfoGain / (double) validModels;

            minResult = 0;
            for (i = 0; i < data.numAttributes(); i++) {
                if ((i != (data).classIndex())
                        && (currentModel[i].checkModel())) // Use 1E-3 here to get a closer approximation to the original
                // implementation.
                {
                    if ((currentModel[i].infoGain() >= (averageInfoGain - 1E-3))
                            && Utils.gr(currentModel[i].gainRatio(), minResult)) {
                        bestModel = currentModel[i];
                        minResult = currentModel[i].gainRatio();
                    }
                }
            }

            if (Utils.eq(minResult, 0)) {
                return noSplitModel;
            }

            bestModel.distribution().
                    addInstWithUnknown(data, bestModel.attIndex());
            Instances m_allData = data;

            if (m_allData != null) {
                bestModel.setSplitPoint(m_allData);
            }
            return bestModel;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public void buildTree(Instances data, boolean keepData) throws Exception {

        Instances[] localInstances;
        pruningInstances = null;
        isLeaf = false;
        isEmpty = false;
        child = null;
        currentClass = null;
        parent = null;

        if (keepData) {
            trainInstances = data;
        }

        if (isSingleClass(data)) {
            isLeaf = true;
            currentClass = data.classAttribute().value(0);

        } else if (isEmptyInstances(data)) {
            currentClass = getMajorityClassinParent();
        } else {
            localInstances = split(data);
            for (int i = 0; i < localInstances.length; i++) {
                buildTree(localInstances[i], keepData);
            }
        }

    }

    public double classifyInstance(Instance instance)
            throws Exception {

        double maxProb = -1;
        double currentProb;
        int maxIndex = 0;
        int j;

        for (j = 0; j < instance.numClasses(); j++) {
            currentProb = 1;//getProbs(j, instance, 1);
            if (Utils.gr(currentProb, maxProb)) {
                maxIndex = j;
                maxProb = currentProb;
            }
        }

        return (double) maxIndex;
    }

    public final void cleanup(Instances justHeaderInfo) {

        trainInstances = justHeaderInfo;
        pruningInstances = null;
        if (!isLeaf) {
            for (int i = 0; i < child.length; i++) {
                child[i].cleanup(justHeaderInfo);
            }
        }
    }

    public double[] distributionForInstance(Instance instance, boolean m_useLaplace) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public String prefix() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public String graph() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public StringBuffer[] toSource(String className) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public String numLeaves() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public String numNodes() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
