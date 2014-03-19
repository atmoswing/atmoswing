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
 
#ifndef ASGEOAREACOMPOSITE_H
#define ASGEOAREACOMPOSITE_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaComposite: public asGeo
{
public:

    /** Default constructor
     * \param coosys The coordinate system
     * \param CornerUL The coordinates of the upper left corner
     * \param CornerUR The coordinates of the upper right corner
     * \param CornerLL The coordinates of the lower left corner
     * \param CornerLR The coordinates of the lower right corner
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaComposite(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    /** Alternative constructor
     * \param coosys The coordinate system
     * \param Umin The left border
     * \param Uwidth The size on U axis
     * \param Vmin The left border
     * \param Vwidth The size on V axis
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaComposite(CoordSys coosys, double Umin, double Uwidth, double Vmin, double Vwidth, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);


    asGeoAreaComposite(CoordSys coosys, float Level = asNONE, float Height = asNONE);


    /** Default destructor */
    virtual ~asGeoAreaComposite();


    void Generate(double Umin, double Uwidth, double Vmin, double Vwidth, int flatAllowed = asFLAT_FORBIDDEN);


    /** Access m_CornerUL
     * \return The current value of m_CornerUL
     */
    Coo GetCornerUL()
    {
        return m_CornerUL;
    }

    /** Access m_CornerUR
     * \return The current value of m_CornerUR
     */
    Coo GetCornerUR()
    {
        return m_CornerUR;
    }

    /** Access m_CornerLL
     * \return The current value of m_CornerLL
     */
    Coo GetCornerLL()
    {
        return m_CornerLL;
    }

    /** Access m_CornerLR
     * \return The current value of m_CornerLR
     */
    Coo GetCornerLR()
    {
        return m_CornerLR;
    }

    /** Access m_Level
     * \return The current value of m_Level
     */
    double GetLevel()
    {
        return m_Level;
    }

    /** Set m_Level
     * \param val New value to set
     */
    void SetLevel(double val)
    {
        m_Level = val;
    }

    /** Access m_AbsoluteUmin
     * \return The current value of m_AbsoluteUmin
     */
    double GetAbsoluteUmin()
    {
        return m_AbsoluteUmin;
    }

    /** Access m_AbsoluteUmax
     * \return The current value of m_AbsoluteUmax
     */
    double GetAbsoluteUmax()
    {
        return m_AbsoluteUmax;
    }

    /** Access m_AbsoluteVmin
     * \return The current value of m_AbsoluteVmin
     */
    double GetAbsoluteVmin()
    {
        return m_AbsoluteVmin;
    }

    /** Access m_AbsoluteVmax
     * \return The current value of m_AbsoluteVmax
     */
    double GetAbsoluteVmax()
    {
        return m_AbsoluteVmax;
    }

    /** Gives the area absolute U width
     * \return The area absolute U width
     */
    double GetAbsoluteUwidth()
    {
        return abs(m_AbsoluteUmax-m_AbsoluteUmin);
    }

    /** Gives the area absolute V width
     * \return The area absolute V width
     */
    double GetAbsoluteVwidth()
    {
        return abs(m_AbsoluteVmax-m_AbsoluteVmin);
    }

    /** Gives the area U min coordinate
     * \return The value of the minimum on the U axis
     */
    double GetUmin();

    /** Gives the area U max coordinate
     * \return The value of the maximum on the U axis
     */
    double GetUmax();

    /** Gives the area V min coordinate
     * \return The value of the minimum on the V axis
     */
    double GetVmin();

    /** Gives the area V max coordinate
     * \return The value of the maximum on the V axis
     */
    double GetVmax();

    /** Gives the area center coordinates
     * \return The coordinates of the center
     */
    Coo GetCenter();

    /** Gives the number of composite areas
     * \return The number of composite areas
     */
    int GetNbComposites()
    {
        return m_NbComposites;
    }

    /** Gives the composite areas
     * \return The composite areas
     */
    vector <asGeoArea> GetComposites()
    {
        return m_Composites;
    }

    /** Gives a specific composite area
     * \return The composite area desired
     */
    asGeoArea GetComposite(int Id)
    {
        if(Id>=m_NbComposites) asThrowException(_("The composite area doesn't exist."));
        return m_Composites[Id];
    }

    /** Tells if the area is a straight rectangle or not
     * \return True if the area is a straight rectangle
     */
    bool IsRectangle();

    /** Convert projection
     * \param newcoordsys The destination projection
     */
    void ProjConvert(CoordSys newcoordsys);

protected:
    /** Creates the composites */
    void CreateComposites();

private:
    double m_AbsoluteUmin;
    double m_AbsoluteUmax;
    double m_AbsoluteVmin;
    double m_AbsoluteVmax;
    Coo m_CornerUL; //!< Member variable "m_CornerUL"
    Coo m_CornerUR; //!< Member variable "m_CornerUR"
    Coo m_CornerLL; //!< Member variable "m_CornerDL"
    Coo m_CornerLR; //!< Member variable "m_CornerDR"
    float m_Level; //!< Member variable "m_Level" hPa
    float m_Height; //!< Member variable "m_Height" m
    int m_NbComposites; //!< Member variable "m_NbComposites"
    vector <asGeoArea> m_Composites; //!< Member variable "m_Composites"
    int m_FlatAllowed; //!< Member variable "m_FlatAllowed"

    /** Process to initialization and checks */
    void Init();

    /** Check every point
     * \return True on success
     */
    bool DoCheckPoints();

    /** Check the consistency of the points
     * \return True on success
     */
    bool CheckConsistency();
};

#endif
