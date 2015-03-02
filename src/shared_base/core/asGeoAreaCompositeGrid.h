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
 
#ifndef asGeoAreaCompositeGrid_H
#define asGeoAreaCompositeGrid_H

#include <asIncludes.h>
#include <asGeoAreaComposite.h>

class asGeoAreaCompositeGrid: public asGeoAreaComposite // Abstract class
{
public:
    enum GridType //!< Define available parameters sets (for the GUI)
    {
        Regular,
        GaussianT62
    };

    asGeoAreaCompositeGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeGrid(float Level = asNONE, float Height = asNONE);


    static asGeoAreaCompositeGrid* GetInstance(const wxString &type, double Xmin, int Xptsnb, double Xstep, double Ymin, int Yptsnb, double Ystep, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    virtual bool GridsOverlay(asGeoAreaCompositeGrid *otherarea) = 0;


    GridType GetGridType()
    {
        return m_gridType;
    }

    wxString GetGridTypeString()
    {
        switch (m_gridType)
        {
            case (Regular):
                return "Regular";
            case (GaussianT62):
                return "GaussianT62";
            default:
                return "Not found";
        }
    }

    virtual double GetXstep() = 0;
    virtual double GetYstep() = 0;

    /** Get the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The axis built on the boundaries and the step
     */
    virtual Array1DDouble GetXaxisComposite(int compositeNb) = 0;

    /** Get the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The axis built on the boundaries and the step
     */
    virtual Array1DDouble GetYaxisComposite(int compositeNb) = 0;

    /** Get the size of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The size of the axis
     */
    virtual int GetXaxisCompositePtsnb(int compositeNb) = 0;

    /** Get the size of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The size of the axis
     */
    virtual int GetYaxisCompositePtsnb(int compositeNb) = 0;

    /** Get the width of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The width of the axis
     */
    virtual double GetXaxisCompositeWidth(int compositeNb) = 0;

    /** Get the width of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The width of the axis
     */
    virtual double GetYaxisCompositeWidth(int compositeNb) = 0;

    /** Get the start value of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The start value of the axis
     */
    virtual double GetXaxisCompositeStart(int compositeNb) = 0;

    /** Get the start value of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The start value of the axis
     */
    virtual double GetYaxisCompositeStart(int compositeNb) = 0;

    /** Get the last value of the X axis for the given composite
     * \param compositeNb The desired composite
     * \return The last value of the axis
     */
    virtual double GetXaxisCompositeEnd(int compositeNb) = 0;

    /** Get the last value of the Y axis for the given composite
     * \param compositeNb The desired composite
     * \return The last value of the axis
     */
    virtual double GetYaxisCompositeEnd(int compositeNb) = 0;

    /** Get the total size of the X axis
     * \return The total size of the axis
     */
    int GetXaxisPtsnb();

    /** Get the total size of the Y axis
     * \return The total size of the axis
     */
    int GetYaxisPtsnb();

    /** Get the total width of the X axis
     * \return The total width of the axis
     */
    double GetXaxisWidth();

    /** Get the total width of the Y axis
     * \return The total width of the axis
     */
    double GetYaxisWidth();

    Array1DDouble GetXaxis();

    Array1DDouble GetYaxis();


protected:
    GridType m_gridType;

private:

};

#endif // asGeoAreaCompositeGrid_H
