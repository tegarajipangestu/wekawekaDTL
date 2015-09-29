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
import weka.classifiers.trees.Id3;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
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

    public String getMajorityClassinParent() {
        Instances instances = parent.trainInstances;
        int numClassAttribute = instances.numDistinctValues(instances.classIndex());
        instances.sort(instances.classIndex());

        return null;
    }

    public String getMajorityClass(Instances _instances) {
        int indexClass = _instances.classIndex();
        Instances instances = _instances;
        instances.sort(indexClass);
        List<String> className = new ArrayList<String>();
        className.add(instances.instance(0).attribute(indexClass).value(0));
        instances.delete(0);

        while (instances.enumerateInstances().hasMoreElements())
        {
            if (instances.instance(0).attribute(indexClass).value(0)==instances.instance(0).attribute(indexClass).value(1))
            {
                
            }
        }
        return null;
    }

    public MyJ48ClassifierTree getParent() {
        return parent;
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
}
