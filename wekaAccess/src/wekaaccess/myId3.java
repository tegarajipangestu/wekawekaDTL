/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaaccess;

import java.util.ArrayList;
import java.util.Collections;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

import java.util.Enumeration;
import java.util.List;
import java.util.Vector;


/**
 *
 * @author Ivana Clairine
 */
  public class myId3 
  extends Classifier {

  /** for serialization */
  static final long serialVersionUID = -2693678647096322561L;
  
  /** The node's successors. */ 
  private myId3[] child;

  /** Attribute used for splitting. */
  private Attribute split_attribute;

  /** Class value if node is leaf. */
  private double leaf_class;

  /** Class distribution if node is leaf. */
  private double[] leaf_distribution;

  /** Class attribute of dataset. */
  private Attribute class_attribute;

  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }

  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    makeTree(data);
  }
  
  public int maxAttr(Instances data, Attribute atr)
  {
      Instances[] maxAttr = splitData(data, atr);
      int[] maxval = new int[atr.numValues()];
      for(int i = 0; i < data.numInstances(); i++)
      {
          Instance temp = data.instance(i);
          maxval[(int) temp.classValue()]++;
      }
      return findmax(maxval);
  }
 
  public int findmax (int[] input)
  {
    int max = -1;
    for (int counter = 1; counter < input.length; counter++)
    {
       if (input[counter] > max)
       {
           max = counter;
       }
    }
    return max;
  }
  
  private void makeTree(Instances data) throws Exception {
    // Check if no instances have reached this node.
    if (data.numInstances() == 0) {
      split_attribute = null;
      leaf_class = Double.NaN;
      leaf_distribution = new double[data.numClasses()];
      return;
    }
    
    // Compute attribute with maximum information gain.
    double[] infoGains = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      infoGains[att.index()] = computeInfoGain(data, att);
    }
    split_attribute = data.attribute(Utils.maxIndex(infoGains));
    
    // Make leaf if information gain is zero. 
    // Otherwise create successors.
    if (Utils.eq(infoGains[split_attribute.index()], 0)) {
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
      child = new myId3[split_attribute.numValues()];
      for (int j = 0; j < split_attribute.numValues(); j++) {
        child[j] = new myId3();
        child[j].makeTree(splitData[j]);
        if(Utils.eq(splitData[j].numInstances(),0))
        {
            child[j].leaf_class = maxAttr(data, data.attribute(j));
        }
      }
    }
  }

  public double classifyInstance(Instance instance) 
    throws Exception {

    if (instance.hasMissingValue()) {
      throw new Exception ("Can't handle missing value(s)");
    }
    if (split_attribute == null) {
      return leaf_class;
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

  private double computeInfoGain(Instances data, Attribute att) 
    throws Exception {

    double infoGain = hitungEntropy(data);
    Instances[] splitData = splitData(data, att);
    for (int j = 0; j < att.numValues(); j++) {
      if (splitData[j].numInstances() > 0) {
        infoGain -= ((double) splitData[j].numInstances() /
                     (double) data.numInstances()) *
          hitungEntropy(splitData[j]);
      }
    }
    return infoGain;
  }
  
   private double hitungEntropy(Instances data) {
    double[] kelas = new double[data.numClasses()];
    for(int i =0; i < data.numInstances(); i++)
    {
       Instance temp = data.instance(i);
       kelas[(int) temp.classValue()]++;
    }
    for(int i = 0; i < data.numClasses(); i++)
    {
        kelas[i] = kelas[i] / data.numInstances();
    }
    double entropi = 0;
    for(int i = 0; i <data.numClasses(); i++)
    {
        if(kelas[i] > 0)
        {
            entropi = entropi - (kelas[i] * Utils.log2(kelas[i]));
        }        
    }
      return entropi;
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

  public static void main(String[] args) {
    runClassifier(new myId3(), args);
  }
}

