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

    public String getMajorityClass(Instances instances) {
        int indexClass = instances.classIndex();
        Instances localInstances = instances;
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
            //Class Majority in parent

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
            currentProb = getProbs(j, instance, 1);
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

    public final double[] distributionForInstance(Instance instance,
            boolean useLaplace)
            throws Exception {

        double[] doubles = new double[instance.numClasses()];

        for (int i = 0; i < doubles.length; i++) {
            if (!useLaplace) {
                doubles[i] = getProbs(i, instance, 1);
            } else {
                doubles[i] = getProbsLaplace(i, instance, 1);
            }
        }

        return doubles;
    }

    /**
     * Assigns a unique id to every node in the tree.
     *
     * @param lastID the last ID that was assign
     * @return the new current ID
     */
    public int assignIDs(int lastID) {

        int currLastID = lastID + 1;

        idNode = currLastID;
        if (child != null) {
            for (int i = 0; i < child.length; i++) {
                currLastID = child[i].assignIDs(currLastID);
            }
        }
        return currLastID;
    }

    /**
     * Returns the type of graph this classifier represents.
     *
     * @return Drawable.TREE
     */
    public int graphType() {
        return Drawable.TREE;
    }

    /**
     * Returns tree in prefix order.
     *
     * @throws Exception if something goes wrong
     * @return the prefix order
     */
    public String prefix() throws Exception {

        StringBuffer text;

        text = new StringBuffer();
        if (isLeaf) {
            text.append("[" + m_localModel.dumpLabel(0, trainInstances) + "]");
        } else {
            prefixTree(text);
        }

        return text.toString();
    }

    /**
     * Returns source code for the tree as an if-then statement. The class is
     * assigned to variable "p", and assumes the tested instance is named "i".
     * The results are returned as two stringbuffers: a section of code for
     * assignment of the class, and a section of code containing support code
     * (eg: other support methods).
     *
     * @param className the classname that this static classifier has
     * @return an array containing two stringbuffers, the first string
     * containing assignment code, and the second containing source for support
     * code.
     * @throws Exception if something goes wrong
     */
    public StringBuffer[] toSource(String className) throws Exception {

        StringBuffer[] result = new StringBuffer[2];
        if (isLeaf) {
            result[0] = new StringBuffer("    p = "
                    + m_localModel.distribution().maxClass(0) + ";\n");
            result[1] = new StringBuffer("");
        } else {
            StringBuffer text = new StringBuffer();
            StringBuffer atEnd = new StringBuffer();

            long printID = MyJ48ClassifierTree.nextID();

            text.append("  static double N")
                    .append(Integer.toHexString(m_localModel.hashCode()) + printID)
                    .append("(Object []i) {\n")
                    .append("    double p = Double.NaN;\n");

            text.append("    if (")
                    .append(m_localModel.sourceExpression(-1, trainInstances))
                    .append(") {\n");
            text.append("      p = ")
                    .append(m_localModel.distribution().maxClass(0))
                    .append(";\n");
            text.append("    } ");
            for (int i = 0; i < child.length; i++) {
                text.append("else if (" + m_localModel.sourceExpression(i, trainInstances)
                        + ") {\n");
                if (child[i].isLeaf) {
                    text.append("      p = "
                            + m_localModel.distribution().maxClass(i) + ";\n");
                } else {
                    StringBuffer[] sub = child[i].toSource(className);
                    text.append(sub[0]);
                    atEnd.append(sub[1]);
                }
                text.append("    } ");
                if (i == child.length - 1) {
                    text.append('\n');
                }
            }

            text.append("    return p;\n  }\n");

            result[0] = new StringBuffer("    p = " + className + ".N");
            result[0].append(Integer.toHexString(m_localModel.hashCode()) + printID)
                    .append("(i);\n");
            result[1] = text.append(atEnd);
        }
        return result;
    }

    /**
     * Returns number of leaves in tree structure.
     *
     * @return the number of leaves
     */
    public int numLeaves() {

        int num = 0;
        int i;

        if (isLeaf) {
            return 1;
        } else {
            for (i = 0; i < child.length; i++) {
                num = num + child[i].numLeaves();
            }
        }

        return num;
    }

    /**
     * Returns number of nodes in tree structure.
     *
     * @return the number of nodes
     */
    public int numNodes() {

        int no = 1;
        int i;

        if (!isLeaf) {
            for (i = 0; i < child.length; i++) {
                no = no + child[i].numNodes();
            }
        }

        return no;
    }

    /**
     * Prints tree structure.
     *
     * @return the tree structure
     */
    public String toString() {

        try {
            StringBuffer text = new StringBuffer();

            if (isLeaf) {
                text.append(": ");
                text.append(m_localModel.dumpLabel(0, trainInstances));
            } else {
                dumpTree(0, text);
            }
            text.append("\n\nNumber of Leaves  : \t" + numLeaves() + "\n");
            text.append("\nSize of the tree : \t" + numNodes() + "\n");

            return text.toString();
        } catch (Exception e) {
            return "Can't print classification tree.";
        }
    }

    protected MyJ48ClassifierTree getNewTree(Instances data) throws Exception {

        MyJ48ClassifierTree newTree = new MyJ48ClassifierTree(m_toSelectModel);
        newTree.buildTree(data, false);

        return newTree;
    }

    protected MyJ48ClassifierTree getNewTree(Instances train, Instances test)
            throws Exception {

        MyJ48ClassifierTree newTree = new MyJ48ClassifierTree(m_toSelectModel);
        newTree.buildTree(train, test, false);

        return newTree;
    }

    private void dumpTree(int depth, StringBuffer text)
            throws Exception {

        int i, j;

        for (i = 0; i < child.length; i++) {
            text.append("\n");;
            for (j = 0; j < depth; j++) {
                text.append("|   ");
            }
            text.append(m_localModel.leftSide(trainInstances));
            text.append(m_localModel.rightSide(i, trainInstances));
            if (child[i].isLeaf) {
                text.append(": ");
                text.append(m_localModel.dumpLabel(i, trainInstances));
            } else {
                child[i].dumpTree(depth + 1, text);
            }
        }
    }

    private void prefixTree(StringBuffer text) throws Exception {

        text.append("[");
        text.append(m_localModel.leftSide(trainInstances) + ":");
        for (int i = 0; i < child.length; i++) {
            if (i > 0) {
                text.append(",\n");
            }
            text.append(m_localModel.rightSide(i, trainInstances));
        }
        for (int i = 0; i < child.length; i++) {
            if (child[i].isLeaf) {
                text.append("[");
                text.append(m_localModel.dumpLabel(i, trainInstances));
                text.append("]");
            } else {
                child[i].prefixTree(text);
            }
        }
        text.append("]");
    }

    /**
     * Help method for computing class probabilities of a given instance.
     *
     * @param classIndex the class index
     * @param instance the instance to compute the probabilities for
     * @param weight the weight to use
     * @return the laplace probs
     * @throws Exception if something goes wrong
     */
    private double getProbsLaplace(int classIndex, Instance instance, double weight)
            throws Exception {

        double prob = 0;

        if (isLeaf) {
            return weight * localModel().classProbLaplace(classIndex, instance, -1);
        } else {
            int treeIndex = localModel().whichSubset(instance);
            if (treeIndex == -1) {
                double[] weights = localModel().weights(instance);
                for (int i = 0; i < child.length; i++) {
                    if (!son(i).isEmpty) {
                        prob += son(i).getProbsLaplace(classIndex, instance,
                                weights[i] * weight);
                    }
                }
                return prob;
            } else {
                if (son(treeIndex).isEmpty) {
                    return weight * localModel().classProbLaplace(classIndex, instance,
                            treeIndex);
                } else {
                    return son(treeIndex).getProbsLaplace(classIndex, instance, weight);
                }
            }
        }
    }

    /**
     * Help method for computing class probabilities of a given instance.
     *
     * @param classIndex the class index
     * @param instance the instance to compute the probabilities for
     * @param weight the weight to use
     * @return the probs
     * @throws Exception if something goes wrong
     */
    private double getProbs(int classIndex, Instance instance, double weight)
            throws Exception {

        double prob = 0;

        if (isLeaf) {
            return weight * localModel().classProb(classIndex, instance, -1);
        } else {
            int treeIndex = localModel().whichSubset(instance);
            if (treeIndex == -1) {
                double[] weights = localModel().weights(instance);
                for (int i = 0; i < child.length; i++) {
                    if (!son(i).isEmpty) {
                        prob += son(i).getProbs(classIndex, instance,
                                weights[i] * weight);
                    }
                }
                return prob;
            } else {
                if (son(treeIndex).isEmpty) {
                    return weight * localModel().classProb(classIndex, instance,
                            treeIndex);
                } else {
                    return son(treeIndex).getProbs(classIndex, instance, weight);
                }
            }
        }
    }

    /**
     * Method just exists to make program easier to read.
     */
    private ClassifierSplitModel localModel() {

        return (ClassifierSplitModel) m_localModel;
    }

    /**
     * Method just exists to make program easier to read.
     */
    private MyJ48ClassifierTree son(int index) {

        return (MyJ48ClassifierTree) child[index];
    }

    /**
     * Returns the revision string.
     *
     * @return	the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10256 $");
    }
}
