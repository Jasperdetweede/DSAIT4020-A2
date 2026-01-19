import numpy as np
import pandas as pd

class ElectricalCircuit:
	def __init__( self ):
		print( "Basic Electrical Circuit Simulator/Generator")

	def gen_predefined_circuits( self, resistances, source_voltage, threshold ):
		if len( resistances ) != 8:
			print( f"Invalid amount of resistors for circuit ({len(resistances)}) out of 8 expected\nDefault values will be used")
			return None

		self.Ra = resistances[0]
		self.Rb = resistances[1]
		self.Rc = resistances[2]
		self.Rd = resistances[3]
		self.Re = resistances[4]
		self.Rf = resistances[5]
		self.Rg = resistances[6]
		self.Rh = resistances[7]
		self.source_voltage = source_voltage
		self.thresholds = threshold
		self.calculate_values()
		return self.get_train_values(), self.is_light_on()

	def gen_random_samples( self, sample_count, thresholds=None, random_state=42 ):
		rng = np.random.default_rng( random_state )
		self.source_voltage = np.round( 10 * rng.random( sample_count ), decimals=2 )
		self.Ra = rng.integers( low=10, high=1000, size=sample_count )
		self.Rb = rng.integers( low=10, high=1000, size=sample_count )
		self.Rc = rng.integers( low=10, high=1000, size=sample_count )
		self.Rd = rng.integers( low=10, high=1000, size=sample_count )
		self.Re = rng.integers( low=10, high=1000, size=sample_count )
		self.Rf = rng.integers( low=10, high=1000, size=sample_count )
		self.Rg = rng.integers( low=10, high=1000, size=sample_count )
		self.Rh = rng.integers( low=10, high=1000, size=sample_count )
		if thresholds is not None:
			self.thresholds = thresholds
		else:
			self.thresholds = np.round( rng.random( sample_count ) / 40, decimals = 4 )
		self.calculate_values()
		return self.get_train_values(), self.is_light_on()

	def calculate_values( self ):
		Rde = ( self.Rd + self.Re ) / ( self.Rd * self.Re )
		Rcde = self.Rc + Rde
		Rbcde = ( Rcde + self.Rb ) / ( Rcde * self.Rb )
		Rfg = ( self.Rf + self.Rg ) / ( self.Rf * self.Rg )
		total_resistance = Rbcde + Rfg + self.Rg

		total_current = self.source_voltage / total_resistance

		self.Ia = total_current
		self.Ib = total_current * Rcde / ( self.Rb + Rcde )
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
		self.Vf = self.If * self.Rf
		self.Vg = self.Ig * self.Rg
		self.Vh = self.Ih * self.Rh

	def is_light_on( self ):
		return pd.DataFrame( self.If * self.Vf > self.thresholds )

	def get_train_values( self ):
		vals = [
			self.source_voltage,
			self.Va,
			self.Ib,
			self.Vc,
			self.Id,
			self.Rd/( self.Ie + 1e-8 ),
			self.Rf/( self.Ig + 1e-8 ),
			self.Vg * self.Ig,
			self.Vh,
			self.thresholds
		]
		return pd.DataFrame( vals )

	def get_embedding( self ):
		return pd.DataFrame( [ self.Ra, self.Rb, self.Rc, self.Rd, self.Re, self.Rf, self.Rg, self.Rh ] )

if __name__ == "__main__":
	ec = ElectricalCircuit()
	X, y = ec.gen_random_samples( 100000 )
	print( y.sum() )
