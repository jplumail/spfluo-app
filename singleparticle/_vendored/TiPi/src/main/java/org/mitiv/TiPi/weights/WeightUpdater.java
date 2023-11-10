/**
 *
 */
package org.mitiv.TiPi.weights;

import org.mitiv.TiPi.array.ShapedArray;

/**
 * @author ferreol
 *
 */
public abstract interface WeightUpdater {
    /**
     * Run specific code
     * @param caller   calling object
     */
    public abstract ShapedArray update(Object caller);

    public abstract ShapedArray getWeights();
}
