/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 * 
 * When distributing Covered Code, include this CDDL Header Notice in 
 * each file and include the License file (licence.txt). If applicable, 
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 * 
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#ifndef asGeoAreaCompositeGaussianGrid_H
#define asGeoAreaCompositeGaussianGrid_H

#include <asIncludes.h>
#include <asGeoAreaCompositeGrid.h>
#include <asGeoAreaGaussianGrid.h>

class asGeoAreaCompositeGaussianGrid: public asGeoAreaCompositeGrid
{
public:

    /** Default constructor
     * \param coosys The coordinate system
     * \param CornerUL The coordinates of the upper left corner
     * \param CornerUR The coordinates of the upper right corner
     * \param CornerLL The coordinates of the lower left corner
     * \param CornerLR The coordinates of the lower right corner
     * \param Xstep The step according to the X axis
     * \param Ystep The step according to the Y axis
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaCompositeGaussianGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, double Xstep, double Ystep, asGeoAreaGaussianGrid::GaussianGridType type=asGeoAreaGaussianGrid::T62, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    /** Alternative constructor
     * \param coosys The coordinate system
     * \param Xmin The left border
     * \param Xptsnb The size on X axis
     * \param Xstep The step according to the X axis
     * \param Ymin The left border
     * \param Yptsnb The size on Y axis
     * \param Ystep The step according to the Y axis
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaCompositeGaussianGrid(double Xmin, int Xptsnb, double Ymin, int Yptsnb, asGeoAreaGaussianGrid::GaussianGridType type=asGeoAreaGaussianGrid::T62, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    /** Default destructor */
    virtual ~asGeoAreaCompositeGaussianGrid();


    bool GridsOverlay(asGeoAreaCompositeGrid *otherarea);

    asGeoAreaGaussianGrid::GaussianGridType GetGaussianGridType()
    {
        return m_gaussianGridType;
    }

    /** Access m_xstep
     * \return The current value of m_xstep
     */
    double GetXstep()
    {
        return 0.0;
    }

    /** Access m_ystep
     * \return The current value of m_ystep
     */
    double GetYstep()
    {
        return 0.0;
    }

    /** Get the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The axis built on the boundaries and the step
     */
    Array1DDouble GetXaxisComposite(int compositeNb);

    /** Get the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The axis built on the boundaries and the step
     */
    Array1DDouble GetYaxisComposite(int compositeNb);

    /** Get the size of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The size of the axis
     */
    int GetXaxisCompositePtsnb(int compositeNb);

    /** Get the size of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The size of the axis
     */
    int GetYaxisCompositePtsnb(int compositeNb);

    /** Get the width of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The width of the axis
     */
    double GetXaxisCompositeWidth(int compositeNb);

    /** Get the width of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The width of the axis
     */
    double GetYaxisCompositeWidth(int compositeNb);

    /** Get the start value of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The start value of the axis
     */
    double GetXaxisCompositeStart(int compositeNb);

    /** Get the start value of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The start value of the axis
     */
    double GetYaxisCompositeStart(int compositeNb);

    /** Get the last value of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The last value of the axis
     */
    double GetXaxisCompositeEnd(int compositeNb);

    /** Get the last value of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The last value of the axis
     */
    double GetYaxisCompositeEnd(int compositeNb);


protected:

private:
    asGeoAreaGaussianGrid::GaussianGridType m_gaussianGridType;
    Array1DDouble m_fullAxisX;
    Array1DDouble m_fullAxisY;

    bool IsOnGrid(const Coo &point);
    bool IsOnGrid(double Xcoord, double Ycoord);
};

#endif // asGeoAreaCompositeGaussianGrid_H
