import numpy as np

class ElectricalCircuit:
	def __init__( self, resistances, source_voltage, threshold ):
		if len( resistances ) != 8:
			print( f"Invalid amount of resistors for circuit ({len(resistances)}) out of 8 expected\nDefault values will be used")
			resistances = 100 * np.ones(8)

		self.Ra = resistances[0]
		self.Rb = resistances[1]
		self.Rc = resistances[2]
		self.Rd = resistances[3]
		self.Re = resistances[4]
		self.Rf = resistances[5]
		self.Rg = resistances[6]
		self.Rh = resistances[7]

		self.source_voltage = source_voltage
		self.calculated = False
		self.threshold = threshold
	
	def __init__( self, random_state, threshold ):
		self.source_voltage = np.random()
		self.resistors = np.random()
		self.calculated = True
		self.threshold = threshold
	
	def calculate_values( self ):
		Rde = ( self.Rd + self.Re ) / ( self.Rd * self.Re )
		Rcde = self.Rc + Rde
		Rbcde = ( self.Rcde + self.Rb ) / ( self.Rcde * self.Rb )
		Rfg = ( self.Rf + self.Rg ) / ( self.Rf * self.Rg )
		total_resistance = Rbcde + Rfg + self.Rg

		total_current = self.source_voltage / total_resistance

		self.Ia = total_current
		self.Ib = total_current * self.Rcde / ( self.Rb + self.Rcde )
		self.Ic = total_current - self.Ib
		self.Id = self.Ic * self.Re / ( self.Rd + self.Re )
		self.Ie = self.Ic - self.Id
		self.If = total_current * self.Rg / ( self.Rf + self.Rg )
		self.Ig = total_current - self.If
		self.Ih = total_current

		self.Va = self.Ia * self.Ra
		self.Vb = self.Ib * self.Rb
		self.Vc = self.Ic * self.Rc
		self.Vd = self.Id * self.Rd
		self.Ve = self.Ie * self.Re
		self.Vd = self.If * self.Rf
		self.Vg = self.Ig * self.Rg
		self.Vh = self.Ih * self.Rh

		self.calculated = True
		
	def is_light_on( self, threshold ):
		if not self.calculated:
			print( "Please use method \"calculate_values\" to simulate the circute first" )
			return None
		return self.If * self.Vf > self.threshold

	def get_train_values( self ):
		vals = [
			self.source_voltage,
			self.Va,
			self.Ib,
			self.Vc,
			self.Id,
			self.Rd/self.Ie,
			self.Rf/self.Ig,
			self.Vg * self.Ig,
			self.Vh,
			self.threshold
		]
		return np.ndarray( vals )

class DCSource:
	def __init__( self, voltage, parent = None, children = None ):
		V = voltage
		I = None
		parent = parent
		children = children

class Resistor:
	def __init__( self, resistance, parent = None, children = None ):
		R = resistance
