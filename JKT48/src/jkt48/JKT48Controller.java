/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package jkt48;

import weka.core.Instances;

/**
 *
 * @author tegar
 */
public class JKT48Controller {
    
    private JKT48Tree treeRoot;

    public JKT48Controller(JKT48Tree treeRoot) {
        this.treeRoot = treeRoot;
    }
    
    public JKT48Controller (Instances i)
    {
        this.treeRoot = new JKT48Tree(i);
    }
    
    public JKT48Tree getTree()
    {
        return treeRoot;
    }
    
    public void buildTree()
    {
        
    }
    
    public void doPruning()
    {
        
    }
    
}
