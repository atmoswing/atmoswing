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
 
#ifndef asGeoAreaGaussianGrid_H
#define asGeoAreaGaussianGrid_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaGaussianGrid: public asGeoArea
{
public:
	enum GaussianGridType
	{
		T62
	};

    /** Default constructor
     * \param coosys The coordinate system
     * \param CornerUL The coordinates of the upper left corner
     * \param CornerUR The coordinates of the upper right corner
     * \param CornerLL The coordinates of the lower left corner
     * \param CornerLR The coordinates of the lower right corner
     * \param type The Gaussian grid type
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaGaussianGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, GaussianGridType type = T62, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    /** Alternative constructor
     * \param coosys The coordinate system
     * \param Umin The left border
     * \param Uptsnb The size on U axis
     * \param Vmin The left border
     * \param Vptsnb The size on V axis
     * \param type The Gaussian grid type
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaGaussianGrid(CoordSys coosys, double Umin, int Uptsnb, double Vmin, int Vptsnb, GaussianGridType type = T62, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_ALLOWED);

    /** Default destructor */
    virtual ~asGeoAreaGaussianGrid();

    /** Get the size of the U axis
     * \return The size of the U axis
     */
    int GetUaxisPtsnb();

    /** Get the size of the V axis
     * \return The size of the V axis
     */
    int GetVaxisPtsnb();

    /** Get the U axis
     * \return The axis built on the boundaries and the step
     */
    Array1DDouble GetUaxis();

    /** Get the V axis
     * \param step The step of the desired axis
     * \return The axis built on the boundaries and the step
     */
    Array1DDouble GetVaxis();

protected:

private:
    Array1DDouble m_FullAxisU;
    Array1DDouble m_FullAxisV;

    bool IsOnGrid(const Coo &point);
    bool IsOnGrid(double Ucoord, double Vcoord);

};

#endif // asGeoAreaGaussianGrid_H
