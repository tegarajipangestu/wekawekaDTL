/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    J48.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.classifiers.trees;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.myj48.BinC45ModelSelection;
import weka.classifiers.trees.myj48.C45ModelSelection;
import weka.classifiers.trees.myj48.ClassifierTree;
import weka.classifiers.trees.myj48.ModelSelection;
import weka.classifiers.trees.myj48.PruneableClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.trees.myj48.MyJ48ClassifierTree;

/**
 * <!-- globalinfo-start -->
 * Class for generating a pruned or unpruned C4.5 decision tree. For more
 * information, see<br/>
 * <br/>
 * Ross Quinlan (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann
 * Publishers, San Mateo, CA.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;book{Quinlan1993,
 *    address = {San Mateo, CA},
 *    author = {Ross Quinlan},
 *    publisher = {Morgan Kaufmann Publishers},
 *    title = {C4.5: Programs for Machine Learning},
 *    year = {1993}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * Valid options are:
 * <p/>
 *
 * <pre> -U
 *  Use unpruned tree.</pre>
 *
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 *
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 *
 * <pre> -R
 *  Use reduced error pruning.</pre>
 *
 * <pre> -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)</pre>
 *
 * <pre> -B
 *  Use binary splits only.</pre>
 *
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 *
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 *
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 *
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 *
 * <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.9 $
 */
public class MyJ48
        extends Classifier
        implements OptionHandler, Drawable, Matchable, Sourcable,
        WeightedInstancesHandler, Summarizable,
        TechnicalInformationHandler {

    /**
     * for serialization
     */
    static final long serialVersionUID = -217733168393644444L;

    /**
     * The decision tree
     */
    private MyJ48ClassifierTree m_root;

    /**
     * Unpruned tree?
     */
    private boolean m_unpruned = false;

    /**
     * Confidence level
     */
    private float m_CF = 0.25f;

    /**
     * Minimum number of instances
     */
    private int m_minNumObj = 2;

    /**
     * Determines whether probabilities are smoothed using Laplace correction
     * when predictions are generated
     */
    private boolean m_useLaplace = false;

    /**
     * Use reduced error pruning?
     */
    private boolean m_reducedErrorPruning = false;

    /**
     * Number of folds for reduced error pruning.
     */
    private int m_numFolds = 3;

    /**
     * Binary splits on nominal attributes?
     */
    private boolean m_binarySplits = false;

    /**
     * Subtree raising to be performed?
     */
    private boolean m_subtreeRaising = true;

    /**
     * Cleanup after the tree has been built.
     */
    private boolean m_noCleanup = false;

    /**
     * Random number seed for reduced-error pruning.
     */
    private int m_Seed = 1;

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {

        return "Class for generating a pruned or unpruned C4.5 decision tree. For more "
                + "information, see\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.BOOK);
        result.setValue(Field.AUTHOR, "Ross Quinlan");
        result.setValue(Field.YEAR, "1993");
        result.setValue(Field.TITLE, "C4.5: Programs for Machine Learning");
        result.setValue(Field.PUBLISHER, "Morgan Kaufmann Publishers");
        result.setValue(Field.ADDRESS, "San Mateo, CA");

        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances the data to train the classifier with
     * @throws Exception if classifier can't be built successfully
     */
    public void buildClassifier(Instances instances)
            throws Exception {

        ModelSelection modSelection;

//    if (m_binarySplits)
//      modSelection = new BinC45ModelSelection(m_minNumObj, instances);
//    else
        modSelection = new C45ModelSelection(m_minNumObj, instances);
//    if (!m_reducedErrorPruning)
        m_root = new MyJ48ClassifierTree();
//    else
//      m_root = new PruneableClassifierTree(modSelection, !m_unpruned, m_numFolds,
//					   !m_noCleanup, m_Seed);
        m_root.buildClassifier(instances);
//    if (m_binarySplits) {
//      ((BinC45ModelSelection)modSelection).cleanup();
//    } else {
        ((C45ModelSelection) modSelection).cleanup();
//    }
    }

    public double classifyInstance(Instance instance) throws Exception {

        return m_root.classifyInstance(instance);
    }

    public final double[] distributionForInstance(Instance instance)
            throws Exception {

        return m_root.distributionForInstance(instance, m_useLaplace);
    }

    public int graphType() {
        return Drawable.TREE;
    }

    public String graph() throws Exception {

        return m_root.graph();
    }

    public String prefix() throws Exception {

        return m_root.prefix();
    }

    public String toSource(String className) throws Exception {

        StringBuffer[] source = m_root.toSource(className);
        return "class " + className + " {\n\n"
                + "  public static double classify(Object[] i)\n"
                + "    throws Exception {\n\n"
                + "    double p = Double.NaN;\n"
                + source[0] // Assignment code
                + "    return p;\n"
                + "  }\n"
                + source[1] // Support code
                + "}\n";
    }

    public Enumeration listOptions() {

        Vector newVector = new Vector(9);

        newVector.
                addElement(new Option("\tUse unpruned tree.",
                                "U", 0, "-U"));
        newVector.
                addElement(new Option("\tSet confidence threshold for pruning.\n"
                                + "\t(default 0.25)",
                                "C", 1, "-C <pruning confidence>"));
        newVector.
                addElement(new Option("\tSet minimum number of instances per leaf.\n"
                                + "\t(default 2)",
                                "M", 1, "-M <minimum number of instances>"));
        newVector.
                addElement(new Option("\tUse reduced error pruning.",
                                "R", 0, "-R"));
        newVector.
                addElement(new Option("\tSet number of folds for reduced error\n"
                                + "\tpruning. One fold is used as pruning set.\n"
                                + "\t(default 3)",
                                "N", 1, "-N <number of folds>"));
        newVector.
                addElement(new Option("\tUse binary splits only.",
                                "B", 0, "-B"));
        newVector.
                addElement(new Option("\tDon't perform subtree raising.",
                                "S", 0, "-S"));
        newVector.
                addElement(new Option("\tDo not clean up after the tree has been built.",
                                "L", 0, "-L"));
        newVector.
                addElement(new Option("\tLaplace smoothing for predicted probabilities.",
                                "A", 0, "-A"));
        newVector.
                addElement(new Option("\tSeed for random data shuffling (default 1).",
                                "Q", 1, "-Q <seed>"));

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     *
     * <!-- options-start -->
     * Valid options are:
     * <p/>
     *
     * <pre> -U
     *  Use unpruned tree.</pre>
     *
     * <pre> -C &lt;pruning confidence&gt;
     *  Set confidence threshold for pruning.
     *  (default 0.25)</pre>
     *
     * <pre> -M &lt;minimum number of instances&gt;
     *  Set minimum number of instances per leaf.
     *  (default 2)</pre>
     *
     * <pre> -R
     *  Use reduced error pruning.</pre>
     *
     * <pre> -N &lt;number of folds&gt;
     *  Set number of folds for reduced error
     *  pruning. One fold is used as pruning set.
     *  (default 3)</pre>
     *
     * <pre> -B
     *  Use binary splits only.</pre>
     *
     * <pre> -S
     *  Don't perform subtree raising.</pre>
     *
     * <pre> -L
     *  Do not clean up after the tree has been built.</pre>
     *
     * <pre> -A
     *  Laplace smoothing for predicted probabilities.</pre>
     *
     * <pre> -Q &lt;seed&gt;
     *  Seed for random data shuffling (default 1).</pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        // Other options
        String minNumString = Utils.getOption('M', options);
        if (minNumString.length() != 0) {
            m_minNumObj = Integer.parseInt(minNumString);
        } else {
            m_minNumObj = 2;
        }
        m_binarySplits = Utils.getFlag('B', options);
        m_useLaplace = Utils.getFlag('A', options);

        // Pruning options
        m_unpruned = Utils.getFlag('U', options);
        m_subtreeRaising = !Utils.getFlag('S', options);
        m_noCleanup = Utils.getFlag('L', options);
        if ((m_unpruned) && (!m_subtreeRaising)) {
            throw new Exception("Subtree raising doesn't need to be unset for unpruned tree!");
        }
        m_reducedErrorPruning = Utils.getFlag('R', options);
        if ((m_unpruned) && (m_reducedErrorPruning)) {
            throw new Exception("Unpruned tree and reduced error pruning can't be selected "
                    + "simultaneously!");
        }
        String confidenceString = Utils.getOption('C', options);
        if (confidenceString.length() != 0) {
            if (m_reducedErrorPruning) {
                throw new Exception("Setting the confidence doesn't make sense "
                        + "for reduced error pruning.");
            } else if (m_unpruned) {
                throw new Exception("Doesn't make sense to change confidence for unpruned "
                        + "tree!");
            } else {
                m_CF = (new Float(confidenceString)).floatValue();
                if ((m_CF <= 0) || (m_CF >= 1)) {
                    throw new Exception("Confidence has to be greater than zero and smaller "
                            + "than one!");
                }
            }
        } else {
            m_CF = 0.25f;
        }
        String numFoldsString = Utils.getOption('N', options);
        if (numFoldsString.length() != 0) {
            if (!m_reducedErrorPruning) {
                throw new Exception("Setting the number of folds"
                        + " doesn't make sense if"
                        + " reduced error pruning is not selected.");
            } else {
                m_numFolds = Integer.parseInt(numFoldsString);
            }
        } else {
            m_numFolds = 3;
        }
        String seedString = Utils.getOption('Q', options);
        if (seedString.length() != 0) {
            m_Seed = Integer.parseInt(seedString);
        } else {
            m_Seed = 1;
        }
    }

    public String[] getOptions() {

        String[] options = new String[14];
        int current = 0;

        if (m_noCleanup) {
            options[current++] = "-L";
        }
        if (m_unpruned) {
            options[current++] = "-U";
        } else {
            if (!m_subtreeRaising) {
                options[current++] = "-S";
            }
            if (m_reducedErrorPruning) {
                options[current++] = "-R";
                options[current++] = "-N";
                options[current++] = "" + m_numFolds;
                options[current++] = "-Q";
                options[current++] = "" + m_Seed;
            } else {
                options[current++] = "-C";
                options[current++] = "" + m_CF;
            }
        }
        if (m_binarySplits) {
            options[current++] = "-B";
        }
        options[current++] = "-M";
        options[current++] = "" + m_minNumObj;
        if (m_useLaplace) {
            options[current++] = "-A";
        }

        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    public String seedTipText() {
        return "The seed used for randomizing the data "
                + "when reduced-error pruning is used.";
    }

    public int getSeed() {

        return m_Seed;
    }

    public void setSeed(int newSeed) {

        m_Seed = newSeed;
    }

    public String useLaplaceTipText() {
        return "Whether counts at leaves are smoothed based on Laplace.";
    }

    public boolean getUseLaplace() {

        return m_useLaplace;
    }

    public void setUseLaplace(boolean newuseLaplace) {

        m_useLaplace = newuseLaplace;
    }

    public String toString() {

        if (m_root == null) {
            return "No classifier built";
        }
        if (m_unpruned) {
            return "J48 unpruned tree\n------------------\n" + m_root.toString();
        } else {
            return "J48 pruned tree\n------------------\n" + m_root.toString();
        }
    }

    public String toSummaryString() {

        return "Number of leaves: " + m_root.numLeaves() + "\n"
                + "Size of the tree: " + m_root.numNodes() + "\n";
    }

    public String measureTreeSize() {
        return m_root.numNodes();
    }

    public String measureNumLeaves() {
        return m_root.numLeaves();
    }

    public String measureNumRules() {
        return m_root.numLeaves();
    }

    public Enumeration enumerateMeasures() {
        Vector newVector = new Vector(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");
        return newVector.elements();
    }

    public String unprunedTipText() {
        return "Whether pruning is performed.";
    }

    public boolean getUnpruned() {

        return m_unpruned;
    }

    public void setUnpruned(boolean v) {

        if (v) {
            m_reducedErrorPruning = false;
        }
        m_unpruned = v;
    }

    public String confidenceFactorTipText() {
        return "The confidence factor used for pruning (smaller values incur "
                + "more pruning).";
    }

    public float getConfidenceFactor() {

        return m_CF;
    }

    public void setConfidenceFactor(float v) {

        m_CF = v;
    }

    public String minNumObjTipText() {
        return "The minimum number of instances per leaf.";
    }

    public int getMinNumObj() {

        return m_minNumObj;
    }

    public void setMinNumObj(int v) {

        m_minNumObj = v;
    }

    public String reducedErrorPruningTipText() {
        return "Whether reduced-error pruning is used instead of C.4.5 pruning.";
    }

    public boolean getReducedErrorPruning() {

        return m_reducedErrorPruning;
    }

    public void setReducedErrorPruning(boolean v) {

        if (v) {
            m_unpruned = false;
        }
        m_reducedErrorPruning = v;
    }

    public String numFoldsTipText() {
        return "Determines the amount of data used for reduced-error pruning. "
                + " One fold is used for pruning, the rest for growing the tree.";
    }

    public int getNumFolds() {

        return m_numFolds;
    }

    public void setNumFolds(int v) {

        m_numFolds = v;
    }

    public String binarySplitsTipText() {
        return "Whether to use binary splits on nominal attributes when "
                + "building the trees.";
    }

    public boolean getBinarySplits() {

        return m_binarySplits;
    }

    public void setBinarySplits(boolean v) {

        m_binarySplits = v;
    }

    public String subtreeRaisingTipText() {
        return "Whether to consider the subtree raising operation when pruning.";
    }

    public boolean getSubtreeRaising() {

        return m_subtreeRaising;
    }

    public void setSubtreeRaising(boolean v) {

        m_subtreeRaising = v;
    }

    public String saveInstanceDataTipText() {
        return "Whether to save the training data for visualization.";
    }

    public boolean getSaveInstanceData() {

        return m_noCleanup;
    }

    public void setSaveInstanceData(boolean v) {

        m_noCleanup = v;
    }


    public static void main(String[] argv) {
        runClassifier(new J48(), argv);
    }
}
