/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jkt48;

import java.util.LinkedList;
import java.util.List;
import weka.core.Instances;

/**
 *
 * @author tegar
 */
public class JKT48Tree {
    
    Instances data;
    JKT48Tree parent;
    List<JKT48Tree> children;

    /**
     *
     * @param data
     */
    public JKT48Tree(Instances data) {
        this.data = data;
        this.children = new LinkedList<JKT48Tree>();
    }
    
    public JKT48Tree() {
        this.data = null;
        this.children = new LinkedList<JKT48Tree>();
    }

    public JKT48Tree addChild(Instances child) {
        JKT48Tree childNode = new JKT48Tree(child);
        childNode.parent = this;
        this.children.add(childNode);
        return childNode;
    }

}
